from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


# Adjust the model to get a higher performance
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        in_size = x.size(0)
        x = self.relu(self.mp(self.conv1(x)))
        x = self.relu(self.mp(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(in_size, -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.logsoftmax(x)


def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    plt.figure()
    pic = None
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx in (1,2,3,4,5):
            if batch_idx == 1:
                pic = data[0,0,:,:]
            else:
                pic = torch.cat((pic,data[0,0,:,:]),dim=1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_fn = nn.NLLLoss()
        loss = loss_fn(output, target)
        # Calculate gradients
        loss.backward()
        # Optimize the parameters according to the calculated gradients
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    plt.imshow(pic.cpu(), cmap='gray')
    plt.show()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_fn = nn.NLLLoss()
            loss = loss_fn(output, target).item()
            test_loss += loss  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # batch_size is a crucial hyper-parameter
    train_kwargs = {'batch_size': args.batch_size} ## 64
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        # Adjust num worker and pin memory according to your computer performance
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Normalize the input (black and white image)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) ##0.1307 和 0.3081 分别是图像像素值的均值和标准差，用减去均值再除以标准差的方式对原图进行标准化
        ])

    # Make train dataset split
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    # Make test dataset split
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    # Convert the dataset to dataloader, including and test_kwargs
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)  ## 64
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs) ## 64

    # Put the model on the GPU or CPU
    model = Net().to(device)

    # Create optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Create a schedule for the optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Begin training and testing
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # Save the model
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()