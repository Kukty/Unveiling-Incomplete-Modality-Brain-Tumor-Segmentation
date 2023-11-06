# loss function for training
import torch
from torch import nn, Tensor
import torch.nn.functional as F

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def softmax_holder_loss(input_logits, target_logits,a=2.0):
    """
    Implement of Holder divergence between two softmax function 
    """
    assert input_logits.size() == target_logits.size()
    p = F.softmax(input_logits, dim=1)
    q = F.softmax(target_logits, dim=1)
    b=a/(a-1)
    # return F.kl_div(input_log_softmax, target_softmax)
    # kl = -torch.log((score_D_u * (1 - score_R)+(1 - score_D_u) * score_R)/ (torch.pow (input=( torch.pow(input=(score_D_u),exponent=a) 
    #         + torch.pow(input=(1-score_D_u),exponent=a)),exponent=1/a ) * torch.pow (input=( torch.pow(input=(score_R),exponent=b) 
    #         + torch.pow(input=(1-score_R),exponent=b)),exponent=1/b)))
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    holder = -torch.log((p*q).sum()/((p**a).sum()**(1/a)*(q**b).sum()**(1/b)))
    return holder

class levelsetLoss(nn.Module):
    def __init__(self):
        super(levelsetLoss, self).__init__()

    def forward(self, output, target):
        # input size = batch x 1 (channel) x height x width
        outshape = output.shape
        tarshape = target.shape
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:,ich], 1)
            target_ = target_.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pcentroid = torch.sum(target_ * output, (2,3))/torch.sum(output, (2,3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss
		
class gradientLoss2d(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss2d, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if(self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW

        loss = torch.sum(dH) + torch.sum(dW)
        return loss
    
class Loss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.recon_loss_2 = torch.nn.MSELoss().cuda()
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0
        self.norm_pix_loss = args.norm_pix_loss
        self.mask_ratio = args.mask_ratio

    def __call__(
        self,
        output_recons,
        target_recons,
        mask,
    ):
        B, C, H, W, D = output_recons.shape
        target_recons = target_recons.reshape(B, C, -1)

        if self.norm_pix_loss:
            mean = target_recons.mean(dim=-1, keepdim=True)
            var = target_recons.var(dim=-1, keepdim=True)
            target_recons = (target_recons - mean) / (var + 1.0e-6) ** 0.5
        target_recons = target_recons.reshape(B, C, H, W, D)
        # masked voxels.
        mask = mask.to(dtype=target_recons.dtype)[None, ...]
        target_recons, output_recons = [val * mask for val in [target_recons, output_recons]]
        recon_loss = self.recon_loss_2(output_recons, target_recons) / self.mask_ratio
        recon_loss = self.alpha3 * recon_loss
        return recon_loss
