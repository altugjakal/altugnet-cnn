import os
import torch
from torch import nn, save, load
from torch.optim import Adam
from colorama import Fore
from adversarial import add_noise


def train_classifier(classifier, train_loader, device, epochs=20, model_path='model_state_cifar.pt'):
    """
    Train the image classifier model.
    
    Args:
        classifier: The classifier model to train
        train_loader: DataLoader for training data
        device: Device to train on (CPU or GPU)
        epochs: Number of training epochs (default: 20)
        model_path: Path to save/load the model (default: 'model_state_cifar.pt')
    """
    if not os.path.exists(model_path):
        optimizer = Adam(classifier.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        loss_list = []

        for epoch in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = classifier(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            loss_list.append(loss.item())
            print(Fore.YELLOW + f"Epoch:{epoch} loss is {loss.item()}")
            if epoch > 0:
                print(Fore.YELLOW + f"Comparing old loss to the new one: {loss_list[epoch - 1] - loss_list[epoch]}")
            print(Fore.YELLOW + '--------------------------------------')

        torch.save(classifier.state_dict(), model_path)
    else:
        with open(model_path, 'rb') as f:
            classifier.load_state_dict(load(f))


def train_denoiser(denoiser, classifier, train_loader, device, epochs=20, model_path='model_state_denoiser.pt'):
    """
    Train the denoiser model.
    
    Args:
        denoiser: The denoiser model to train
        classifier: Classifier model (used to generate adversarial examples)
        train_loader: DataLoader for training data
        device: Device to train on (CPU or GPU)
        epochs: Number of training epochs (default: 20)
        model_path: Path to save/load the model (default: 'model_state_denoiser.pt')
    """
    if not os.path.exists(model_path):
        optimizer = Adam(denoiser.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        loss_list = []

        for epoch in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                images.requires_grad = True
                optimizer.zero_grad()

                noised_images = add_noise(images, labels, classifier)
                predicted_clean = denoiser(noised_images)
                loss = loss_fn(predicted_clean, images)

                loss.backward()
                optimizer.step()

            loss_list.append(loss.item())
            print(Fore.LIGHTBLUE_EX + f"Epoch:{epoch} loss is {loss.item()}")
            if epoch > 0:
                print(Fore.LIGHTBLUE_EX + f"Comparing old loss to the new one: {loss_list[epoch - 1] - loss_list[epoch]}")
            print(Fore.LIGHTBLUE_EX + '--------------------------------------')

        torch.save(denoiser.state_dict(), model_path)
    else:
        with open(model_path, 'rb') as f:
            denoiser.load_state_dict(load(f))