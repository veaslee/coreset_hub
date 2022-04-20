import sys
import numpy as np
import torch
from torch.utils.data import random_split
import torchvision
import random
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

data_path = '/project/Leaves/datasets'

transforms_cifar_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])

transforms_cifar_noaug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])

transforms_mnist_aug = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])

transforms_mnist_noaug = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])

transforms_tinyimagenet_aug = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])

transforms_tinyimagenet_noaug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])

def load_cifar(dataset='cifar10', batch_size=128, DataAug=False):

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    if DataAug:
        transforms_train = transforms_cifar_aug
    else:
        transforms_train = transforms_cifar_noaug
    
    transforms_test = transforms_cifar_noaug

    if dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms_train)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transforms_test)
        num_classes = 10
    elif dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_train)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transforms_test)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes


def load_noise_cifar(dataset='cifar10', batch_size=128, DataAug=False, label_noise_ratio=0):

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    if DataAug:
        transforms_train = transforms_cifar_aug
    else:
        transforms_train = transforms_cifar_noaug
    
    transforms_test = transforms_cifar_noaug

    if dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms_train)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transforms_test)
        num_classes = 10
    elif dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_train)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transforms_test)
        num_classes = 100

    random.seed(9)
    random_label_num = int(label_noise_ratio * len(trainset.targets))
    random_label_list = [random.randint(0, num_classes - 1) for i in range(random_label_num)]
    trainset.targets = [random_label_list[index] if index < len(random_label_list)  else label for index, label in enumerate(trainset.targets)]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes


def load_two_testset_cifar(dataset='cifar10'):

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    
    transforms_test = transforms_cifar_noaug

    if dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transforms_test)
    elif dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transforms_test)

    test_set, val_set = random_split(dataset=testset, lengths=[5000,5000], generator=torch.Generator().manual_seed(9))
    testloader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=2)
    return testloader, valloader

def load_mnist(batch_size=128, DataAug=False):

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    
    if DataAug:
        transforms_train = transforms_mnist_aug
    else:
        transforms_train = transforms_mnist_noaug
    
    transforms_test = transforms_mnist_noaug

    trainset = datasets.MNIST(data_path, train=True, download=True,
                       transform=transforms_train)
    testset = datasets.MNIST(data_path, train=False,
                       transform=transforms_test)
    num_classes = 10

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes


def load_fashionmnist(batch_size=128, DataAug=False, random_label=False):

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    if DataAug:
        transforms_train = transforms_mnist_aug
    else:
        transforms_train = transforms_mnist_noaug
    
    transforms_test = transforms_mnist_noaug

    trainset = datasets.FashionMNIST(data_path, train=True, download=True,
                       transform=transforms_train)
    testset = datasets.FashionMNIST(data_path, train=False,
                       transform=transforms_test)
    num_classes = 10

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes


class Dataset():
    def __init__(self, x, y, transform=None):
        assert(len(x) == len(y))
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform is not None:
            x = self.transform(Image.fromarray(x.astype(np.uint8)))
        return x, y

    def __len__(self):
        return len(self.x)

def half_TinyImageNet(root=data_path, train=True, sub='former', transform=None):
    if train:
        path = '{}/tiny-imagenet/train.npz'.format(root)
    else:
        path = '{}/tiny-imagenet/test.npz'.format(root)

    data = np.load(path)
    num_classes = 200
    x = data['images']
    y = data['labels']
    if sub == 'former':
        x = data['images'][data['labels']<(num_classes//2)]
        y = data['labels'][data['labels']<(num_classes//2)]
    elif sub == 'latter':
        x = data['images'][data['labels']>(num_classes//2 - 1)]
        y = data['labels'][data['labels']>(num_classes//2 - 1)] - num_classes//2

    return Dataset(x=x, y=y, transform=transform)


def load_half_tinyimagenet(batch_size=128, DataAug=False, sub='former', random_label=False):
    if DataAug:
        transforms_train = transforms_tinyimagenet_aug
    else:
        transforms_train = transforms_tinyimagenet_noaug
    
    transforms_test = transforms_tinyimagenet_noaug

    trainset = half_TinyImageNet(data_path, train=True, sub=sub, transform=transforms_train)
    testset = half_TinyImageNet(data_path, train=False, sub=sub, transform=transforms_test)
    num_classes = 200 // 2

    if random_label:
        random.shuffle(trainset.targets)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes


def TinyImageNet(root=data_path, train=True, transform=None):
    if train:
        path = '{}/tiny-imagenet/train.npz'.format(root)
    else:
        path = '{}/tiny-imagenet/test.npz'.format(root)

    data = np.load(path)

    return Dataset(x=data['images'], y=data['labels'], transform=transform)


def load_tinyimagenet(batch_size=128, DataAug=False, random_label=False):
    if DataAug:
        transforms_train = transforms_tinyimagenet_aug
    else:
        transforms_train = transforms_tinyimagenet_noaug
    
    transforms_test = transforms_tinyimagenet_noaug

    trainset = TinyImageNet(data_path, train=True, transform=transforms_train)
    testset = TinyImageNet(data_path, train=False, transform=transforms_test)

    num_classes = 200

    if random_label:
        random.shuffle(trainset.targets)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes

def load_half_cifar(dataset='cifar10', batch_size=128, DataAug=False, sub='former', random_label=False):

    # Data Uplaod
    if DataAug:
        transforms_train = transforms_cifar_aug
    else:
        transforms_train = transforms_cifar_noaug
    
    transforms_test = transforms_cifar_noaug

    if dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms_train)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transforms_test)
        num_classes = 10
    elif dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_train)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transforms_test)
        num_classes = 100

    trainset.targets = np.array(trainset.targets)
    testset.targets = np.array(testset.targets)

    if sub == 'former':
        trainset.data = trainset.data[trainset.targets<(num_classes//2)]
        testset.data = testset.data[testset.targets<(num_classes//2)]
        trainset.targets = trainset.targets[trainset.targets<(num_classes//2)]
        testset.targets = testset.targets[testset.targets<(num_classes//2)]
        num_classes = num_classes // 2
    elif sub == 'latter':
        trainset.data = trainset.data[trainset.targets>(num_classes//2 - 1)]
        testset.data = testset.data[testset.targets>(num_classes//2 - 1)]
        trainset.targets = trainset.targets[trainset.targets>(num_classes//2 - 1)] - num_classes//2 
        testset.targets = testset.targets[testset.targets>(num_classes//2 - 1)] - num_classes//2
        num_classes = num_classes // 2
    
    if random_label:
        random.shuffle(trainset.targets)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes


def load_noise_input(shape_like='cifar10', len=10000):
    if shape_like == 'mnist':
        size = (len, 28, 28, 1)
        transform = transforms_mnist_noaug
    elif shape_like == 'cifar10' or shape_like == 'cifar100':
        size = (len, 32, 32, 3)
        transform = transforms_cifar_noaug
    elif shape_like == 'tinyimagenet':
        size = (len, 64, 64, 3)
        transform = transforms_tinyimagenet_noaug
    x = np.random.randint(256, size=size)
    y = np.random.randint(10, size=len)
    testset = Dataset(x=x, y=y, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


def load_gan_image_input(dataset):
    if dataset == 'cifar10':
        path='/public/data1/users/leishiye/datasets/cifar10_BigGAN.npz'
    else:
        path='/public/data1/users/leishiye/datasets/cifar100_BigGAN.npz'
    transform = transforms_cifar_noaug
    fake_data = np.load(path)
    x = fake_data['x'].transpose((0, 2, 3, 1))
    y = fake_data['y']
    testset = Dataset(x=x, y=y, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


def load_style_image_input(folder_path='/public/data1/users/leishiye/datasets/styleformer_cifar'):
    transform = transforms_cifar_noaug
    testset = ImageFolder(folder_path, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)


def load_cifar_coreset(dataset='cifar10', batch_size=128, coreset_index=[], frac=1.0):

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')

    if dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms_cifar_aug)
        num_classes = 10
    elif dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_cifar_aug)
        num_classes = 100

    frac_index = np.array(coreset_index[: int(len(coreset_index) * frac)])
    trainset.data = np.array(trainset.data)[frac_index]
    trainset.targets = np.array(trainset.targets)[frac_index]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader, num_classes


def load_cifar_coreset_complement(dataset='cifar10', batch_size=128, coreset_index=[], frac=0.):

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')

    if dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms_cifar_aug)
        num_classes = 10
    elif dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_cifar_aug)
        num_classes = 100

    frac_index = np.array(coreset_index[int(len(coreset_index) * frac):])
    trainset.data = np.array(trainset.data)[frac_index]
    trainset.targets = np.array(trainset.targets)[frac_index]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader, num_classes