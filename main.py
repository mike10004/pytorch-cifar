"""Train CIFAR10 with PyTorch."""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch import Tensor

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import models
from utils import progress_bar


MODEL_FACTORIES = {
    models.VGG.__name__: lambda: models.VGG("VGG19"),
    models.ResNet18.__name__: models.ResNet18,
    models.PreActResNet18.__name__: models.PreActResNet18,
    models.GoogLeNet.__name__: models.GoogLeNet,
    models.DenseNet121.__name__: models.DenseNet121,
    models.ResNeXt29_2x64d.__name__: models.ResNeXt29_2x64d,
    models.MobileNet.__name__: models.MobileNet,
    models.MobileNetV2.__name__: models.MobileNetV2,
    models.DPN92.__name__: models.DPN92,
    models.ShuffleNetG2.__name__: models.ShuffleNetG2,
    models.SENet18.__name__: models.SENet18,
    models.ShuffleNetV2.__name__: lambda: models.ShuffleNetV2(1),
    models.EfficientNetB0.__name__: models.EfficientNetB0,
    models.RegNetX_200MF.__name__: models.RegNetX_200MF,
    models.SimpleDLA.__name__: models.SimpleDLA,
}



def main():
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10 Training',
        epilog=f"""\
Models available:
  {(os.linesep + "  ").join(sorted(MODEL_FACTORIES.keys()))}
""")
    parser.add_argument("model", metavar="NAME", choices=set(MODEL_FACTORIES.keys()))
    parser.add_argument("--epoch-count", type=int, default=200)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    epoch_count = args.epoch_count
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model

    net_name = args.model or models.SimpleDLA.__name__
    print('==> Building model', net_name, "on", device)
    net_factory = MODEL_FACTORIES[net_name]
    net = net_factory()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs: Tensor
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs: Tensor
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc


    for epoch_ in range(start_epoch, start_epoch + epoch_count):
        train(epoch_)
        test(epoch_)
        scheduler.step()


if __name__ == '__main__':
    sys.exit(main())
