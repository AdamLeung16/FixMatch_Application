import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from models.wideresnet import WideResNet
from models.ema import ModelEMA
from dataset.cifar import get_cifar10
import math
import os
import shutil
from torch.optim.lr_scheduler import LambdaLR

BATCH_SIZE = 64
NUM_WORKERS = 4
MU = 7  # 批次中无标签数据与有标签数据的比例
NUM_LABELED = 250   # 有标签数据的数量
TOTAL_STEPS = 2**20 # 总训练步骤数
EVAL_STEP = 1024    # 每个Epoch中的批次训练次数
NUM_CLASSES = 10    # 分类类别数量

best_acc = 0    # 初始化最佳准确率
# 保存权重文件
def save_checkpoint(state, is_best, checkpoint, filename='checkpoint_250.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best: # 如果是最佳模型
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best_250.pth')) # 复制文件为最佳模型文件

# 余弦退火学习率调度器
def get_cosine_schedule(optimizer,num_training_steps,num_cycles=7./16.,last_epoch=-1):
    # 内部函数用于计算学习率
    def _lr_lambda(current_step):
        no_progress = float(current_step) / float(max(1, num_training_steps))   # 计算无进展步数比例
        return max(0., math.cos(math.pi * num_cycles * no_progress))    # 余弦退火减少学习率
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

# 数据交错，用于增强数据
def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# 数据反交错
def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# 定义FixMatch类
class FixMatch:
    def __init__(self, model, ema_model, optimizer, scheduler, epochs, num_classes, threshold=0.95):
        self.model = model  # 主模型
        self.ema_model = ema_model  # EMA（指数移动平均）模型
        self.optimizer = optimizer  # 优化器
        self.scheduler = scheduler  # 学习率调度器
        self.epochs = epochs    # 训练轮数
        self.num_classes = num_classes  # 分类类别数量
        self.threshold = threshold  # 伪标签置信度阈值

    # 模型训练
    def train(self, labeled_trainloader, unlabeled_trainloader, test_loader):
        global best_acc
        test_accs = []  # 存储测试准确率
        labeled_iter = iter(labeled_trainloader)
        unlabeled_iter = iter(unlabeled_trainloader)
        self.model.train()
        for epoch in range(self.epochs):
            for batch_idx in range(EVAL_STEP):
                try:
                    inputs_x, targets_x = labeled_iter.next()
                except:
                    labeled_iter = iter(labeled_trainloader)
                    inputs_x, targets_x = labeled_iter.next()
                try:
                    (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                except:
                    unlabeled_iter = iter(unlabeled_trainloader)
                    (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                batch_size = inputs_x.shape[0]
                inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*MU+1).cuda()   # 将有标签和无标签数据合并并交错
                targets_x = targets_x.cuda()
                logits = self.model(inputs) # 模型预测
                logits = de_interleave(logits, 2*MU+1)  # 反交错
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)   # 无标签数据的弱增强和强增强预测结果
                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean') # 有标签数据的交叉熵损失
                pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)   # 计算无标签数据的伪标签
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)  # 获得伪标签和其最大概率
                mask = max_probs.ge(self.threshold).float() # 置信度大于阈值的伪标签
                Lu = (F.cross_entropy(logits_u_s, targets_u,reduction='none') * mask).mean()    # 无标签数据的损失，仅对高置信度样本计算
                loss = Lx + Lu  # 总损失

                loss.backward() # 反向传播
                self.optimizer.step()   # 更新优化器参数
                self.scheduler.step()   # 更新学习率
                self.ema_model.update(self.model)   # 更新EMA模型
                self.model.zero_grad()  # 清零梯度
                print(f'Epoch {epoch+1}/{self.epochs}, Iter {batch_idx+1}/{EVAL_STEP}, Loss: {loss.item()}')
                
                test_model = self.ema_model.ema
                test_acc = self.test(test_loader, test_model)   # 测试准确率
                is_best = test_acc > best_acc   # 判断当前测试准确率是否为最佳
                best_acc = max(test_acc, best_acc)  # 更新最佳准确率

                model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # 获取需要保存的模型
                ema_to_save = self.ema_model.ema.module if hasattr(self.ema_model.ema, "module") else self.ema_model.ema    # 获取需要保存的EMA模型
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }, is_best, 'result')   # 调用save_checkpoint函数保存模型
                test_accs.append(test_acc)  # 将测试准确率添加到列表中
    
    # 模型测试
    def test(self, test_loader, model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images.cuda())  # 模型预测
                _, predicted = torch.max(outputs.data, 1)   # 获取预测结果
                total += labels.size(0) # 总样本数
                correct += (predicted == labels.cuda()).sum().item()    # 计算正确预测的样本数
        test_acc = correct / total  # 计算测试准确率
        print('Accuracy: %d %%' % (100 * test_acc))
        return test_acc

def main():
    # 数据准备和预处理
    labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10('./data',NUM_LABELED,NUM_CLASSES,BATCH_SIZE,EVAL_STEP)
    # 有标签数据加载器、无标签数据加载器、测试数据加载器
    labeled_trainloader = DataLoader(labeled_dataset,sampler=RandomSampler(labeled_dataset),batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,drop_last=True)
    unlabeled_trainloader = DataLoader(unlabeled_dataset,sampler=RandomSampler(unlabeled_dataset),batch_size=BATCH_SIZE*MU,num_workers=NUM_WORKERS,drop_last=True)
    test_loader = DataLoader(test_dataset,sampler=SequentialSampler(test_dataset),batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)

    # 初始化WideResNet模型
    model = WideResNet(depth=28, widen_factor=2, drop_rate=0.3, num_classes=NUM_CLASSES).cuda()
    
    # 定义模型参数组
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 5e-4},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = optim.SGD(grouped_parameters, lr=0.03,momentum=0.9, nesterov=True)  # 初始化优化器
    epochs = math.ceil(TOTAL_STEPS / EVAL_STEP) # 计算训练的轮数
    scheduler = get_cosine_schedule(optimizer, TOTAL_STEPS) # 获取学习率调度器
    ema_model = ModelEMA(model, decay=0.999)    # 初始化EMA模型

    model.zero_grad()   # 清零梯度

    # 初始化FixMatch模型并开始训练
    myFixMatch = FixMatch(model,ema_model,optimizer,scheduler,epochs,NUM_CLASSES,threshold=0.95)
    myFixMatch.train(labeled_trainloader,unlabeled_trainloader,test_loader)

if __name__=='__main__':
    main()