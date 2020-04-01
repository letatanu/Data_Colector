from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from DB_Loader import DB_Loader
from db_handler import DB_Handler


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=30,kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(30, 60, 2, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(42240, 15)
        # self.fc2 = nn.Linear(100, 15)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 3)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # pred = output.argmax(dim=1, keepdim=True).view_as(target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.cpu()))
    # with torch.no_grad():
    #     test_loss = 0
    #     correct = 0
    #     count = 0
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         count += 1
    #         data, target = data.to(device, dtype=torch.float), target.to(device)
    #         output = model(data)
    #         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #         test_loss += criterion(output, target)  # sum up batch loss
    #         correct += target.eq(pred.view_as(target)).sum().item()
    #     test_loss /= count
    #     print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #         test_loss, correct, len(train_loader.dataset),
    #         100. * correct / len(train_loader.dataset)))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    LossFunc = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device)
            output = model(data)
            test_loss += LossFunc(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += target.eq(pred.view_as(target)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                        help='input batch size for set_1 (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA set_1')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging set_1 status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    #
    # torch.manual_seed(args.seed)

    device = torch.device(0 if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    db_handler = DB_Handler(data_dir="./Data", scene_dir="set_1")
    trainDB_loader = DB_Loader(root_dir="./Data/training/set_1", db=db_handler.read_db(db_set="training"))

    train_loader = torch.utils.data.DataLoader(trainDB_loader,
                                               batch_size=args.batch_size, shuffle=True, **kwargs)

    testDB_loader = DB_Loader(root_dir="./Data/testing/set_1", db=db_handler.read_db(db_set="testing"))
    test_loader = torch.utils.data.DataLoader(testDB_loader, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    try:
        model = Net().to(device)
        model.load_state_dict(torch.load("model_test1"))
    except:
        model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.5)
    # optimizer = optim.Adam(params=model.parameters(), lr=0.9)
    # optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.7)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        if epoch % 5 == 0:
            torch.save(model.state_dict(), "model_test1")


if __name__ == '__main__':
    main()
