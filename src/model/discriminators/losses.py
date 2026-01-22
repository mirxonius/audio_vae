"""
Loss functions for GAN-based audio discriminators.
"""

from typing import List

import torch
import torch.nn.functional as F


def discriminator_loss(
    discriminator_real_logits: List[torch.Tensor],
    discriminator_fake_logits: List[torch.Tensor],
) -> torch.Tensor:
    """
    Hinge loss for discriminator.
    
    Args:
        discriminator_real_logits: List of logits from real samples
        discriminator_fake_logits: List of logits from fake/generated samples
        
    Returns:
        Averaged hinge loss across all discriminators
    """
    loss = 0.0
    for real_logits, fake_logits in zip(
        discriminator_real_logits, discriminator_fake_logits
    ):
        real_loss = torch.mean(F.relu(1.0 - real_logits))
        fake_loss = torch.mean(F.relu(1.0 + fake_logits))
        loss += real_loss + fake_loss
    return loss / len(discriminator_real_logits)


def generator_adversarial_loss(
    disc_fake_outputs: List[torch.Tensor],
) -> torch.Tensor:
    """
    Adversarial loss for generator (hinge formulation).
    
    Args:
        disc_fake_outputs: List of discriminator logits for generated samples
        
    Returns:
        Averaged adversarial loss
    """
    loss = 0.0
    for fake_logits in disc_fake_outputs:
        loss += torch.mean(-fake_logits)
    return loss / len(disc_fake_outputs)


def feature_matching_loss(
    disc_real_fmaps: List[List[torch.Tensor]],
    disc_fake_fmaps: List[List[torch.Tensor]],
) -> torch.Tensor:
    """
    Feature matching loss between real and fake feature maps.
    
    Computes L1 distance between intermediate features from discriminator
    for real and generated samples.
    
    Args:
        disc_real_fmaps: List of feature map lists from real samples
        disc_fake_fmaps: List of feature map lists from fake samples
        
    Returns:
        Averaged feature matching loss
    """
    loss = 0.0
    num_features = 0

    for real_features, fake_features in zip(disc_real_fmaps, disc_fake_fmaps):
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
            num_features += 1

    return loss / num_features if num_features > 0 else loss
