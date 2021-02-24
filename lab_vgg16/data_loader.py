import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from chord import chordDiagram


def clean_loader_cifar(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    clean_loader= DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return clean_loader


def adv_loader_data(args, adv_samples, targets):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    adv_loader = DataLoader(TensorDataset(adv_samples, targets), batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return  adv_loader


def robust_inferece(model, loader, args, target_model=False, note='None'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            try:
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
            except:
                test_loss += F.cross_entropy(output[-1], target, reduction='sum').item()
                pred = output[-1].data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(loader.dataset)
    if target_model:
        sr = 100. * float(correct) / len(loader.dataset)
        psr = sr
    else:
        sr = 100-100. * float(correct) / len(loader.dataset)
        psr = 100-sr
    print('<< {} >> Average loss: {:.4f}, Predict Success Rate: {}/{} ({:.2f}%)'.format(
         note, test_loss, correct, len(loader.dataset), psr))


def robust_evaluate(model, loader, args, target_model=False, note='None'):
    model.eval()

    label_list=[]
    pred_list=[]
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            _, output = model(data)
            pred = output.data.max(1, keepdim=True)[1]

            label_list.extend(target.cpu().numpy().tolist())
            pred_list.extend(pred.squeeze().cpu().numpy().tolist())

    confusion = confusion_matrix(label_list, pred_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print('Confusion Matrix:\n', confusion)

    ax = plt.axes()
    nodePos = chordDiagram(confusion, ax)
    ax.axis('off')
    prop = dict(fontsize=16 * 0.8, ha='center', va='center')
    nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(10):
        ax.text(nodePos[i][0], nodePos[i][1], nodes[i], rotation=nodePos[i][2], **prop)
    #plt.show()
    plt.savefig('vgg16_vallina_pgd.png')

    return confusion
