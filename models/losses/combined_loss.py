"""
Combined Loss Function for DMIAD
Includes reconstruction loss, memory loss, and sparsity loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp


class IntensityLoss(nn.Module):
    def __init__(self):
        super(IntensityLoss, self).__init__()

    def forward(self, prediction, target):
        return torch.mean(torch.pow(torch.abs(prediction - target), 2))


class L2Loss(nn.Module):
    def __init__(self, eps=1e-8):
        super(L2Loss, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        error = torch.mean(torch.pow((prediction - target), 2))
        error = torch.sqrt(error + self.eps)
        return error


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    cs = (cs + 1) / 2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
        ret = (ret + 1) / 2
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
        ret = (ret + 1) / 2

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


def feature_map_permute(input):
    s = input.data.shape
    l = len(s)

    # permute feature channel to the last:
    # NxCxDxHxW --> NxDxHxW x C
    if l == 2:
        x = input # NxC
    elif l == 3:
        x = input.permute(0, 2, 1)
    elif l == 4:
        x = input.permute(0, 2, 3, 1)
    elif l == 5:
        x = input.permute(0, 2, 3, 4, 1)
    else:
        x = []
        print('wrong feature map size')
    x = x.contiguous()
    # NxDxHxW x C --> (NxDxHxW) x C
    x = x.view(-1, s[1])
    return x


class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b


class EntropyLossEncap(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLossEncap, self).__init__()
        self.eps = eps
        self.entropy_loss = EntropyLoss(eps)

    def forward(self, input):
        score = feature_map_permute(input)
        ent_loss_val = self.entropy_loss(score)
        return ent_loss_val


class MemoryLoss(nn.Module):
    """Memory-related losses for dual memory modules"""
    def __init__(self, entropy_weight=0.0002):
        super(MemoryLoss, self).__init__()
        self.entropy_weight = entropy_weight
        self.entropy_loss = EntropyLossEncap()
    
    def forward(self, memory_results):
        """
        Compute memory losses
        
        Args:
            memory_results: Dict containing memory module outputs
        """
        memory_loss = 0.0
        loss_dict = {}
        
        # Temporal memory entropy loss
        if 'temporal_output' in memory_results and 'att' in memory_results['temporal_output']:
            temp_att = memory_results['temporal_output']['att']
            temp_entropy = self.entropy_loss(temp_att)
            memory_loss += self.entropy_weight * temp_entropy
            loss_dict['temporal_entropy'] = temp_entropy
        
        # Spatial memory entropy loss
        if 'spatial_output' in memory_results and memory_results['spatial_output'] is not None:
            if 'att' in memory_results['spatial_output']:
                spa_att = memory_results['spatial_output']['att']
                spa_entropy = self.entropy_loss(spa_att)
                memory_loss += self.entropy_weight * spa_entropy
                loss_dict['spatial_entropy'] = spa_entropy
        
        loss_dict['total_memory_loss'] = memory_loss
        return memory_loss, loss_dict


class SparsityLoss(nn.Module):
    """Sparsity regularization for memory attention"""
    def __init__(self):
        super(SparsityLoss, self).__init__()
    
    def forward(self, memory_results):
        """
        Compute sparsity loss to encourage sparse attention
        """
        sparsity_loss = 0.0
        count = 0
        
        # Temporal memory sparsity
        if 'temporal_output' in memory_results and 'raw_attention' in memory_results['temporal_output']:
            temp_att = memory_results['temporal_output']['raw_attention']
            temp_sparsity = torch.mean(torch.sum(temp_att ** 2, dim=1))
            sparsity_loss += temp_sparsity
            count += 1
        
        # Spatial memory sparsity
        if 'spatial_output' in memory_results and memory_results['spatial_output'] is not None:
            if 'raw_attention' in memory_results['spatial_output']:
                spa_att = memory_results['spatial_output']['raw_attention']
                spa_sparsity = torch.mean(torch.sum(spa_att ** 2, dim=1))
                sparsity_loss += spa_sparsity
                count += 1
        
        if count > 0:
            sparsity_loss = sparsity_loss / count
        
        return sparsity_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for DMIAD
    
    Includes:
    - Reconstruction loss (Intensity + L2 + SSIM)
    - Memory entropy loss 
    - Sparsity regularization
    """
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        
        # Loss weights from config
        self.reconstruction_weight = config.LOSS.RECONSTRUCTION_WEIGHT
        self.memory_weight = config.LOSS.MEMORY_WEIGHT
        self.sparsity_weight = config.LOSS.SPARSITY_WEIGHT
        
        # Individual loss components
        self.intensity_loss = IntensityLoss()
        self.l2_loss = L2Loss()
        self.msssim_loss = MSSSIM()
        self.memory_loss = MemoryLoss()
        self.sparsity_loss = SparsityLoss()
        
        # Loss combination weights
        self.intensity_weight = 1.0
        self.l2_weight = 1.0
        self.msssim_weight = 1.0
    
    def forward(self, input_imgs, reconstructed_imgs, memory_results, labels=None):
        """
        Compute combined loss
        
        Args:
            input_imgs: [B, C, H, W] - Original images
            reconstructed_imgs: [B, C, H, W] - Reconstructed images
            memory_results: Dict containing memory module outputs
            labels: [B] - Image labels (optional, for training)
        
        Returns:
            Dict of loss components
        """
        loss_dict = {}
        
        # === 1. Reconstruction Loss ===
        intensity_loss = self.intensity_loss(reconstructed_imgs, input_imgs)
        l2_loss = self.l2_loss(reconstructed_imgs, input_imgs) 
        msssim_loss = (1 - self.msssim_loss(reconstructed_imgs, input_imgs))/2
        
        reconstruction_loss = (
            self.intensity_weight * intensity_loss +
            self.l2_weight * l2_loss +
            self.msssim_weight * msssim_loss
        )
        
        loss_dict['intensity_loss'] = intensity_loss
        loss_dict['l2_loss'] = l2_loss
        loss_dict['msssim_loss'] = msssim_loss
        loss_dict['reconstruction_loss'] = reconstruction_loss
        
        # === 2. Memory Loss ===
        memory_loss_val, memory_loss_dict = self.memory_loss(memory_results)
        loss_dict.update(memory_loss_dict)
        
        # === 3. Sparsity Loss ===
        sparsity_loss_val = self.sparsity_loss(memory_results)
        loss_dict['sparsity_loss'] = sparsity_loss_val
        
        # === 4. Total Loss ===
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.memory_weight * memory_loss_val +
            self.sparsity_weight * sparsity_loss_val
        )
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


# Example usage and testing
if __name__ == "__main__":
    from types import SimpleNamespace
    
    # Create dummy config
    config = SimpleNamespace()
    config.LOSS = SimpleNamespace()
    config.LOSS.RECONSTRUCTION_WEIGHT = 1.0
    config.LOSS.MEMORY_WEIGHT = 0.01
    config.LOSS.SPARSITY_WEIGHT = 0.0001
    
    # Create loss function
    criterion = CombinedLoss(config)
    
    # Test inputs
    input_imgs = torch.randn(4, 3, 256, 256)
    reconstructed_imgs = torch.randn(4, 3, 256, 256)
    
    # Mock memory results
    memory_results = {
        'temporal_output': {
            'att': torch.randn(4, 2000, 32, 32),
            'raw_attention': torch.softmax(torch.randn(4*32*32, 2000), dim=1)
        },
        'spatial_output': {
            'att': torch.randn(4, 2000, 32, 32),
            'raw_attention': torch.softmax(torch.randn(4*32*32, 2000), dim=1)
        }
    }
    
    # Test forward pass
    loss_dict = criterion(input_imgs, reconstructed_imgs, memory_results)
    
    print("Loss components:")
    for loss_name, loss_value in loss_dict.items():
        print(f"{loss_name}: {loss_value.item():.6f}")
    
    print("\nCombined loss function test completed!")