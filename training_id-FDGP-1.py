import argparse
import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from data_pairs import SiameseNetwork, SiameseNetworkDataset
from loss import ContrastiveLoss
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


def _save_best_model(model, best_loss, epoch, country):
    # Save Model
    model_name = country + '_net'
    state = {
        'state_dict': model.state_dict(),
        'best_acc': best_loss,
        'cur_epoch': epoch
    }
    if not os.path.isdir('./models'):
        os.makedirs('./models')

    torch.save(state, './models/' + model_name + '.ckpt')


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    args = parser.parse_args()

    # train_set_dir for a country
    country = 'alb'
    train_set_dir = "./data/training_set/" + country + "/"

    train_me_where = "from_beginning"  # "from_middle"

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    # Create model
    model = SiameseNetwork().to(device)

    print('Model created.')
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # model = nn.DataParallel(model.cuda())
        model = nn.DataParallel(model.to(device))

    print('model and cuda mixing done')

    # Loss
    criterion_contrastive = ContrastiveLoss()

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    batch_size = args.bs
    best_loss = 1000.0

    # Load data
    folder_dataset = datasets.ImageFolder(root=train_set_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((300, 300)),
                                                                          transforms.ToTensor()]), should_invert=False)
    train_loader = DataLoader(siamese_dataset, shuffle=True, num_workers=1, batch_size=batch_size)
    print("Total number of batches in train loader are :", len(train_loader))

    if train_me_where == "from_middle":
        checkpoint = torch.load('./models/' + country + '_net.ckpt')
        model.load_state_dict(checkpoint['state_dict'])

    writer = SummaryWriter('./runs/real_siamese_net_running')

    print('-' * 10)

    for epoch in range(args.epochs):
        epoch_training_loss = 0.0
        num_batches = 0
        for i, data in enumerate(train_loader, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img0, img1)
            loss_contrastive = criterion_contrastive(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

            epoch_training_loss += loss_contrastive.item()
            num_batches += 1
        print('-' * 5)
        print("epoch: ", epoch, ", loss: ", epoch_training_loss / num_batches)

        if epoch_training_loss / num_batches < best_loss:
            best_loss = epoch_training_loss / num_batches
            print('Saving...')
            _save_best_model(model, best_loss, epoch, country)


if __name__ == '__main__':
    main()
