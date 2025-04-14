import torch
import torch.nn.functional as F
import numpy as np
import math
from monai.losses import GlobalMutualInformationLoss, LocalNormalizedCrossCorrelationLoss, BendingEnergyLoss, DiceLoss

class Dice:
    def __init__(self, **kwargs):
        self.loss_func = DiceLoss(**kwargs)

    def loss(self, y_true, y_pred):
        return self.loss_func(y_pred,y_true) 


class MI:
    def __init__(self, **kwargs):
        self.loss_func = GlobalMutualInformationLoss(**kwargs)

    def loss(self, y_true, y_pred):
        return self.loss_func(y_pred,y_true)

class NCC:
    def __init__(self,**kwargs):
        self.loss_func = LocalNormalizedCrossCorrelationLoss(**kwargs)

    def loss(self, y_true, y_pred):
        return self.loss_func(y_pred,y_true)

class MSE:
    """
    Mean squared error loss. (SSD)
    """
    def loss(self, y_true, y_pred, reduction="mean"):
        return F.mse_loss(y_true,y_pred, reduction=reduction)
class MAE:
    """
    Mean absolute error loss. (SAD)
    """
    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

class BendingEnergy:
    def __init__(self, **kwargs):
        self.loss_func = BendingEnergyLoss(**kwargs)

    def loss(self, _, field):
        return self.loss_func(field)
    

class Det_Jac:
    """
    Class to compute metrics related to Jacobian Detemrinant, can be used for:
        - Loss based (mean or sum for det jac map).
        - Return actual det jac map (return_map argument).
        - Return negative and positive det jac counts (return_counts argument).

    An additional argument, 'weighted' given during the call can be used to weight
    the loss in various ways which we are still under investigation.
    """
    def __init__(self, return_map=False, return_counts=False, weight_type=None):
        assert not (return_map and return_counts), "Setting both return flags to True is currently not supported."
        assert weight_type in [None, "loss_weight", "jacobian_goal"], f"Unrecognized weighting strategy : {weight_type=} "
        self.return_map = return_map
        self.return_counts = return_counts
        self.weight_type = weight_type

    def _computeDetJac2D(self, phi):
        dx = torch.gradient(phi[:,0,:,:], axis=(1,2)) # Warning
        dy = torch.gradient(phi[:,1,:,:], axis=(1,2)) # Warning 
          
        phiX_dx, phiX_dy = dx
        phiY_dx, phiY_dy = dy

        determinant = phiX_dx*phiY_dy - phiY_dx*phiX_dy
        return determinant
    
    def _computeDetJac3D(self, phi):
        # [1, 3, 32, 32, 32])
        dx = torch.gradient(phi[:,0,:,:,:], axis=(1,2,3)) # Warning
        dy = torch.gradient(phi[:,1,:,:,:], axis=(1,2,3)) # Warning 
        dz = torch.gradient(phi[:,2,:,:,:], axis=(1,2,3)) # Warning 
        
        
        phiX_dx, phiX_dy, phiX_dz = dx
        phiY_dx, phiY_dy, phiY_dz = dy
        phiZ_dx, phiZ_dy, phiZ_dz = dz

        plus = (phiX_dx * phiY_dy * phiZ_dz) + (phiX_dy * phiY_dz * phiZ_dx) + (phiX_dz * phiY_dx * phiZ_dy)  
        minus = (phiX_dz * phiY_dy * phiZ_dx) + (phiX_dy * phiY_dx * phiZ_dz) + (phiX_dx * phiY_dz * phiZ_dy)  
        determinant = plus - minus
        return determinant
    

    def loss(self, weight_map, field):
        #TODO changed for reproducibility...
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        inp_shape = field.shape
        assert (inp_shape[1]==2 and (len(inp_shape) == 4) )  or (inp_shape[1]==3 and (len(inp_shape) == 5) ), f" Input shape {inp_shape} incorrect, expected (batch, 2, h, w) for 2D or (batch, 3, d, h, w) for 3D data"
        
        dim = len(inp_shape) - 2
        
        mesh = torch.meshgrid(*[torch.arange(0, s) for s in inp_shape[2::]])
        identity = torch.stack(mesh).unsqueeze(0)#.permute(0,2,3,1)


        if dim == 2 :
            _, _, H, W = field.shape
            # TODO Becuase using monai... : https://docs.monai.io/en/latest/_modules/monai/networks/blocks/warp.html#Warp , line " grid[..., i] = grid[..., i] * 2 / (dim - 1) - 1 "
            grid_x = (field[:,0,:,:] + 1) * (H -1)/2
            grid_y = (field[:,1,:,:] + 1) * (W -1)/2
            phi = torch.stack((grid_x, grid_y), 1)  # shape (N, D, H, W, 3)
            phi = field.to(device) + identity.to(device) #
            determinant = self._computeDetJac2D(phi)


            determinant[:,0,:] = torch.ones_like(determinant[:,0,:])
            determinant[:,:,0] = torch.ones_like(determinant[:,:,0])
        else:
            _, _, H, W, D = field.shape
            grid_x = (field[:,0,:,:,:] + 1) * (D -1)/2
            grid_y = (field[:,1,:,:,:] + 1) * (W -1)/2
            grid_z = (field[:,2,:,:,:] + 1) * (H -1)/2
            phi = torch.stack((grid_x, grid_y, grid_z), 1)#1) #4)  # shape (N, D, H, W, 3)
            # phi = field.to(device) + identity.to(device) #
            phi = phi.to(device) + identity.to(device) #

            
            determinant = self._computeDetJac3D(phi)
            # TODO commented now... # Border effects : 
            determinant[:,0,:,:] = torch.ones_like(determinant[:,0,:,:])
            determinant[:,:,0,:] = torch.ones_like(determinant[:,:,0,:])
            determinant[:,:,:,0] = torch.ones_like(determinant[:,:,:,0])
        # Get difference from ones (volume preserving):
        difference = (determinant - torch.ones(determinant.shape).to(device))
        # If using weights map as loss pixel ponderation:
        if self.weight_type == "loss_weight":
            difference = difference*weight_map

        # Get the actual MSE:
        loss = torch.mean(difference**2)
        # If we only want the determinant (e.g to plot, overwrite the loss)
        if self.return_map :
            loss = determinant
        if self.return_counts:
            num_pos = torch.sum(determinant > 0)
            loss = (num_pos, torch.numel(determinant)- num_pos)
        return loss


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()



