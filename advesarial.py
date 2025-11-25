import torch
from torch import nn


def add_noise(images, labels, classifier, epsilon=0.05):
    """
    Generate adversarial examples using FGSM attack.
    
    Args:
        images: Input images
        labels: True labels
        classifier: Classifier model
        epsilon: Perturbation magnitude (default: 0.05)
    
    Returns:
        Perturbed images
    """
    outputs = classifier(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    perturbed = images + epsilon * torch.sign(images.grad)
    images.grad.zero_()
    return perturbed.detach()


def denoise_images(images, denoiser):
    """
    Denoise images using the denoiser model.
    
    Args:
        images: Noisy input images
        denoiser: Denoiser model
    
    Returns:
        Denoised images
    """
    outputs = denoiser(images)
    return outputs