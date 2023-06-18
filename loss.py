import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    # Contrastive loss function Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_1, output_2, lbl):
        euclidean_distance = F.pairwise_distance(output_1, output_2, keepdim=True)
        contrastive_loss = torch.mean((1 - lbl) * torch.pow(euclidean_distance, 2) +
                                      lbl * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return contrastive_loss
