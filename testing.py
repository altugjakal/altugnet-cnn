import torch
from torch import nn
from colorama import Fore
import matplotlib.pyplot as plt
from adversarial import add_noise, denoise_images
from data_loader import CIFAR10_LABELS


def test_clean_classification(classifier, test_loader, device):
    """
    Test the classifier on clean (unperturbed) images.
    
    Args:
        classifier: The classifier model
        test_loader: DataLoader for test data
        device: Device to run on (CPU or GPU)
    """
    test_loss_list = []
    test_accuracy_list = []
    loss_fn = nn.CrossEntropyLoss()

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = classifier(images)
        loss = loss_fn(outputs, labels)
        test_loss_list.append(loss.item())

        _, predicted = torch.max(outputs, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total
        test_accuracy_list.append(accuracy)

    avg_test_loss = sum(test_loss_list) / len(test_loss_list)
    avg_test_accuracy = sum(test_accuracy_list) / len(test_accuracy_list)
    
    print(Fore.RED + f"Resultant accuracy from the plain test dataset: {avg_test_accuracy}")
    print(Fore.YELLOW + f"Average test loss: {avg_test_loss}")


def test_adversarial_classification(classifier, denoiser, test_loader, device, use_denoiser=False):
    """
    Test the classifier on adversarially perturbed images.
    
    Args:
        classifier: The classifier model
        denoiser: The denoiser model
        test_loader: DataLoader for test data
        device: Device to run on (CPU or GPU)
        use_denoiser: Whether to use the denoiser (default: False)
    """
    second_test_accuracy_list = []
    altered_example_img = None
    non_altered_example_img = None
    pre_noiser_test = None
    post_noiser_test = None

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        original_images = images.clone().detach()
        images.requires_grad = True

        non_altered_example_img = images
        pre_adv_true_label = labels.item()
        
        outputs_OG = classifier(images)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs_OG, labels)

        noised = add_noise(images, labels, classifier)

        if use_denoiser:
            pre_noiser_test = noised
            noised = denoise_images(noised, denoiser)
            post_noiser_test = noised

        altered_example_img = noised
        outputs_adv = classifier(noised)

        adv_total = labels.size(0)
        _, adv_predicted = torch.max(outputs_adv, 1)
        _, actual_predicted = torch.max(outputs_OG, 1)
        post_adv_false_label = adv_predicted.item()
        adv_correct = (adv_predicted == labels).sum().item()
        adv_accuracy = adv_correct / adv_total
        second_test_accuracy_list.append(adv_accuracy)

    avg_adversarial_accuracy = sum(second_test_accuracy_list) / len(second_test_accuracy_list)
    print(Fore.RED + f"Resultant accuracy from the adversarial attack: {avg_adversarial_accuracy} ({CIFAR10_LABELS[pre_adv_true_label]})")
    
    if use_denoiser:
        print(Fore.RED + f"Denoiser in action")
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(pre_noiser_test.squeeze().cpu().detach().permute(1, 2, 0).numpy())
        axs[0].set_title('Noised Input')
        axs[0].axis('off')

        axs[1].imshow(post_noiser_test.squeeze().cpu().detach().permute(1, 2, 0).numpy())
        axs[1].set_title('Denoised Output')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    # Always show original vs attacked image
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(non_altered_example_img.squeeze().cpu().detach().permute(1, 2, 0).numpy())
    axs[0].set_title(f"Original: {CIFAR10_LABELS[actual_predicted.item()]}")
    axs[0].axis('off')

    axs[1].imshow(altered_example_img.squeeze().cpu().detach().permute(1, 2, 0).numpy())
    axs[1].set_title(f"Attacked: {CIFAR10_LABELS[post_adv_false_label]}")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()