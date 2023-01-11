import torch
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict, List



def get_data_loader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=False):
  if dataset == 'fashion-mnist':
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
  elif dataset == 'mnist':
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
  else:
    raise ValueError('Invalid dataset name')
  
  num_train = len(trainset)
  indices = list(range(num_train))
  split = int(0.8 * num_train)
  train_idx, valid_idx = indices[:split], indices[split:]
  train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
  valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

  train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
  valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, pin_memory=pin_memory)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

  return train_loader, valid_loader, test_loader
