import wandb
import os
import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from monai.metrics import DiceMetric
from collections import defaultdict
from src.losses import NCC, Det_Jac, Grad, MSE
from monai.losses import DiceCELoss, DiceLoss

import torch.nn as nn

def save_plot(values, name, values_2=None, label1=None, label2=None):
    plt.plot(values, label=label1)
    if values_2:
        plt.plot(values_2, label=label2)
        plt.legend()
    plt.savefig(name)
    plt.close()

def mask_to_labels(mask):
    out_labels = torch.zeros((mask.shape[-2],mask.shape[-1]))
    for i in range(8):
        out_labels +=mask[i,:,:]*(i+1)
    return out_labels




######### For n inputs :

def forward_n(fixed_image, moving_images, moving_segs, model, warp_layer, amplitudes, fixed_as_input=True,teacher_model=None, amplitudes_wtih_fixed=None):
    """
    Perform a forward pass through the neural network model to predict DDF and warp moving images/labels.

    Args:
        fixed_image: Fixed image tensor.
        moving_images: List of moving image tensors.
        moving_segs: List of moving segmentation mask tensors, in one-hot format.
        model: Neural network model.
        warp_layer: Warp layer used in the model.
        amplitudes: Amplitudes tensor to condition the registration. If None, not used.
        fixed_as_input: Flag indicating whether the fixed image is included in the input to the model (default is True).
        teacher_model: Teacher model used for distillation (default is None).
        amplitudes_wtih_fixed: Amplitudes tensor with leading 0 if required in the input for the teacher model (default is None).

    Returns:
        If teacher_model is None:
            flows: List of flow tensors predicted by the model.
            moved_images: List of warped moving image tensors.
            moved_segs: List of warped moving segmentation tensors.
        If teacher_model is not None:
            flows_teacher: List of flow tensors predicted by the teacher model.
            flows: List of flow tensors predicted by the model.
            moved_images: List of warped moving image tensors.
            moved_segs: List of warped moving segmentation tensors.
    """
    # predict DDF through LocalNet
    input = torch.cat(([fixed_image] + moving_images), dim=1)
    if not fixed_as_input:
        input = torch.cat((moving_images), dim=1)
    # print(f"{input.shape=}")
    if amplitudes is not None:
        raw_flows = model(input, amplitudes).float()
    else:
        raw_flows = model(input).float()
    flows = [raw_flows[:,2*i:(2*i)+2,:,:] for i in range(len(moving_images))]

    # warp moving image and label with the predicted ddf
    moved_images = [warp_layer(moving_images[i], flows[i]) for i in range(len(flows))]
    # warp moving label (optional)
    moved_segs = [warp_layer(moving_segs[i], flows[i]) for i in range(len(flows))]

    if teacher_model is not None:
        with torch.no_grad():
            input = torch.cat(([fixed_image] + moving_images), dim=1)
            raw_flows_teacher = teacher_model(input, amplitudes_wtih_fixed).float()
            flows_teacher = [raw_flows_teacher[:,2*i:(2*i)+2,:,:] for i in range(len(moving_images))]
            return flows_teacher, flows, moved_images, moved_segs
    return flows, moved_images, moved_segs
 
def forward(fixed_image, moving_image, moving_seg, model, warp_layer):
    input = torch.cat(([fixed_image,moving_image]), dim=1)
    # predict DDF 
    flow = model(input).float()
    # warp moving image and label with the predicted ddf
    moved_image = warp_layer(moving_image, flow)
    # warp moving label (optional)
    moved_seg = warp_layer(moving_seg, flow)

    return flow, moved_image, moved_seg

def loss_fun_simple_n(fixed_image, moved_images,
            fixed_seg,moved_segs,
            flows,
            weight_sim,weight_reg,weight_dice,
            device,agreement_weight=0,
            flows_teacher=None, weight_distillation = 0):
    """
    Compute a custom multi-target loss function for registration tasks, with:

    Args:
        fixed_image: Fixed image tensor.
        moved_images: List of moved image tensors.
        fixed_seg: Fixed segmentation mask tensor in one-hot format.
        moved_segs: List of moved segmentation masks tensors mask tensor in one-hot format.
        flows: List of registrationflow tensors used.
        weight_sim: Weight for similarity loss.
        weight_reg: Weight for regularization loss.
        weight_dice: Weight for dice loss.
        device: Device used for computation.
        agreement_weight: Weight for agreement loss (default is 0).
        flows_teacher: List of flow tensors from a teacher model for distillation (default is None).
        weight_distillation: Weight for distillation loss if using a teacher model (default is 0).

    Returns:
        sim_losses: List of similarity loss values for each target.
        reg_losses: List of regularization loss values for each target.
        dice_losses: List of dice loss values for each target.
        total_sim_loss: Total similarity loss.
        total_reg_loss: Total regularization loss.
        total_dice_loss: Total dice loss.
        total_loss: Total combined loss.
        total_agreement_loss: Total agreement loss.
        distillation_losses: List of distillation loss values for each target.
        total_distillation_loss: Total distillation loss.
    """
    # Instantiate where necessary
    if weight_sim > 0:
        sim_loss_func = NCC(spatial_dims=2, kernel_size=5)
    if weight_reg > 0:
        reg_loss_func = Det_Jac()
    if weight_dice > 0:
        dice_loss_func = DiceLoss()
    sim_losses = []
    reg_losses = []
    dice_losses = []
    distillation_losses = []
    total_sim_loss = torch.Tensor([0.0]).to(device)
    total_reg_loss = torch.Tensor([0.0]).to(device)
    total_dice_loss = torch.Tensor([0.0]).to(device)
    total_distillation_loss = torch.Tensor([0.0]).to(device)
    # Compute loss components
    mean_image = torch.zeros_like(moved_images[0])
    for i in range(len(flows)):
        mean_image += moved_images[i]
        sim_loss = sim_loss_func.loss(fixed_image, moved_images[i]) if weight_sim > 0 else torch.Tensor([0.0])
        reg_loss = reg_loss_func.loss(None,flows[i]) if weight_reg > 0 else torch.Tensor([0.0])
        dice_loss = dice_loss_func(fixed_seg, moved_segs[i]) if weight_dice > 0 else torch.Tensor([0.0])
        
        total_sim_loss += sim_loss
        total_reg_loss += reg_loss
        total_dice_loss += dice_loss
        sim_losses.append(sim_loss.item())
        reg_losses.append(reg_loss.item())
        dice_losses.append(dice_loss.item())

        distill_loss = torch.Tensor([0.0]).to(device)
        if weight_distillation > 0 :
            distill_loss = nn.MSELoss()(flows[i],flows_teacher[i])
        total_distillation_loss += distill_loss
        distillation_losses.append(distill_loss.item())

        
    total_agreement_loss = torch.Tensor([0.0]).to(device)
    mean_image /= len(flows)
    if agreement_weight>0:
        MSE_func = MSE()
        total_agreement_loss = MSE_func.loss(fixed_image,mean_image)
        
    # Weighted combination:
    # total_loss = (sim_loss * weight_sim) + (reg_loss * weight_reg) + (dice_loss * weight_dice)
    total_sim_loss /= len(flows)
    total_reg_loss /= len(flows)
    total_dice_loss /= len(flows)
    total_distillation_loss /= len(flows)
    total_loss = (total_sim_loss * weight_sim) + (total_reg_loss * weight_reg) + (total_dice_loss * weight_dice) + (total_agreement_loss * agreement_weight) + (total_distillation_loss*weight_distillation)
    return sim_losses, reg_losses, dice_losses, total_sim_loss,total_reg_loss, total_dice_loss,total_loss, total_agreement_loss, distillation_losses, total_distillation_loss  #total_loss, sim_loss, reg_loss, dice_loss

def loss_fun_simple(fixed_image, moved_image,
            fixed_seg,moved_seg,
            flow,
            weight_sim,weight_reg,weight_dice,
            device):
    """
    Custom multi-target loss:
        - Parametrizable weights for components: NCC/ Similarity (Det JAC)/Dice loss
    Note: Might require "calibration" of lambda weights for the multi-target components,
        e.g. by making a first trial run, and manually setting weights to account for different magnitudes
        - Intra registration Loss (Sim and dice...)
        - TODO :  smarter intra sample loss + additional reg for ranking?
    """
    # Instantiate where necessary
    if weight_sim > 0:
        sim_loss_func = NCC(spatial_dims=2, kernel_size=5)
    if weight_reg > 0:
        reg_loss_func = Det_Jac()
    if weight_dice > 0:
        dice_loss_func = DiceLoss()
    # sim_losses = []
    # reg_losses = []
    # dice_losses = []
    # total_sim_loss = torch.Tensor([0.0]).to(device)
    # total_reg_loss = torch.Tensor([0.0]).to(device)
    # total_dice_loss = torch.Tensor([0.0]).to(device)
    # Compute loss components
    sim_loss = sim_loss_func.loss(fixed_image, moved_image) if weight_sim > 0 else torch.Tensor([0.0])
    reg_loss = reg_loss_func.loss(None,flow) if weight_reg > 0 else torch.Tensor([0.0])
    dice_loss = dice_loss_func(fixed_seg, moved_seg) if weight_dice > 0 else torch.Tensor([0.0])

    total_loss = (sim_loss * weight_sim) + (reg_loss * weight_reg) + (dice_loss * weight_dice)
    return sim_loss, reg_loss, dice_loss, total_loss


def train_loop(args, model,warp_layer,optimizer,lr_scheduler,
               dice_metric_before,dice_metric_after,
               epoch, max_epochs,
               loader, weight_sim,weight_reg,weight_dice,
               metrics,
               best_eval_loss, best_eval_dice, pth_best_loss, pth_best_dice, 
               dir_save,
               device, do_save=True, wandb_log=True,
               num_save_samples=2):
    # ==============================================
    # Train
    # ==============================================
    t0_train = time.time()
    mode = "Training" if model.training else "Validating"
    dict_prefix = "train" if model.training else "val"
    epoch_loss, epoch_sim_loss, epoch_reg_loss, epoch_dice_loss, n_steps= 0, 0, 0, 0, 0
    dice_before,dice_after = [], []
    for batch_data in tqdm.tqdm(loader, desc=f"{mode}..."):
        # if weight_dice >0 :
        fixed_image, fixed_seg = batch_data["fixed_image"].to(device).squeeze(-1), batch_data["fixed_seg"].to(device).squeeze(-1)
        moving_image, moving_seg = batch_data[f"moving_image"].to(device).squeeze(-1), batch_data[f"moving_seg"].to(device).squeeze(-1)
        
        n_steps += fixed_image.shape[0]
        # Forward pass and loss
        if mode == "Training":
            optimizer.zero_grad()
        flow, moved_image, moved_seg = forward(fixed_image, moving_image, moving_seg, model, warp_layer)
        
        (total_sim_loss,total_reg_loss, total_dice_loss,
            total_loss) = loss_fun_simple(fixed_image, moved_image,fixed_seg,moved_seg,flow,
                                            weight_sim,weight_reg,weight_dice,
                                            device)
        if mode == "Training":
            # Optimise
            total_loss.backward()
            optimizer.step()

        epoch_loss += total_loss.item() * fixed_image.shape[0]
        epoch_sim_loss += total_sim_loss.item() * fixed_image.shape[0]
        epoch_reg_loss += total_reg_loss.item() * fixed_image.shape[0]
        epoch_dice_loss += total_dice_loss.item() * fixed_image.shape[0]
        
        # Dice :    
        moved_seg = moved_seg.round()
        moving_seg = moving_seg.round()
        fixed_seg = fixed_seg.round()

        dice_metric_before(y_pred=moving_seg, y=fixed_seg)
        dice_metric_after(y_pred=moved_seg, y=fixed_seg)



            
        ####### NEWWW SHOULD BE DONE PER OUTPUT .........
        dice_before_value = dice_metric_before.aggregate().item()
        dice_metric_before.reset()
        dice_after_value = dice_metric_after.aggregate().item()
        dice_metric_after.reset()
            
        dice_before.append(dice_before_value)
        dice_after.append(dice_after_value)
    if not mode == "Training":
        lr_scheduler.step()
    # Loss
    epoch_loss /= n_steps
    epoch_sim_loss /= n_steps
    epoch_reg_loss /= n_steps
    epoch_dice_loss /= n_steps
    # Conserve metrics:
    metrics[f"{dict_prefix}/loss"].append(epoch_loss)
    metrics[f"{dict_prefix}/sim_loss"].append(epoch_sim_loss)
    metrics[f"{dict_prefix}/reg_loss"].append(epoch_reg_loss)
    metrics[f"{dict_prefix}/dice_loss"].append(epoch_dice_loss)

    # Print Epoch info :
    print(f"{mode} Epoch {epoch + 1}/{max_epochs} : ")
    print(f"___________ TOTAL {mode} INFO ___________")
    print(f"\t loss: {epoch_loss:.3f}")
    detailed_info = f"\t  Sim loss: {epoch_sim_loss:.3f} , Reg loss: {epoch_reg_loss:.3f} ,  Dice loss ={epoch_dice_loss:.3f} "
    print(detailed_info)
    print(f"\t elapsed time: {time.time()-t0_train:.2f} sec.")
    print("__________________________________________________")


    
    # Mean the metrics and print and log
    mean_dice_before = np.mean(dice_before)
    mean_dice_after = np.mean(dice_after)
    print(f"\t ---- For mean : dice_before: {mean_dice_before:.3f}, dice_after: {mean_dice_after:.3f}")
    metrics[f"{dict_prefix}/dice"].append(mean_dice_after)
    metrics[f"{dict_prefix}/dice_before"].append(mean_dice_before)
    
    # Additional validation metrics and save model eventually..:
    if not (mode == "Training"):
        
        if total_loss.item() < best_eval_loss: 
            best_eval_loss = total_loss.item()
            # Save best model based on dice
            if pth_best_loss != "":
                os.remove(os.path.join(dir_save, pth_best_loss))
            pth_best_loss = f"best_total_loss_{epoch + 1}_{best_eval_loss:.3f}.pth"
            torch.save(model.state_dict(), os.path.join(dir_save, pth_best_loss))
            print(f"{epoch + 1} | Saving best TRE model: {pth_best_loss}")

        if mean_dice_after > best_eval_dice:
            print(f"Save change .. {mean_dice_after=} > {best_eval_dice=}")
            best_eval_dice = mean_dice_after
            print(f"After change .. {mean_dice_after=} > {best_eval_dice=}")
            if do_save:
                # Save best model based on Dice
                if pth_best_dice != "":
                    os.remove(os.path.join(dir_save, pth_best_dice))
                pth_best_dice = f"best_dice_{epoch + 1}_{best_eval_dice:.3f}.pth"
                torch.save(model.state_dict(), os.path.join(dir_save, pth_best_dice))
                print(f"{epoch + 1} | Saving best Dice model: {pth_best_dice}")
        if wandb_log:
                # metrics["lr"] = optimizer.param_groups[0]['lr']
                wandb.log({k: v[-1] for k, v in metrics.items()} | {"lr": optimizer.param_groups[0]['lr']})

    return model, metrics, best_eval_loss, pth_best_loss, best_eval_dice, pth_best_dice


def train_loop2(args, model,warp_layer,optimizer,lr_scheduler,
               dice_metric_before,dice_metric_after,
               epoch, max_epochs,
               loader, weight_sim,weight_reg,weight_dice,
               metrics,
               best_eval_loss, best_eval_dice, pth_best_loss, pth_best_dice, 
               dir_save,
               device, do_save=True, wandb_log=True,
               num_moving=10,
               num_save_samples=2,
               teacher_model=None):
    """
    Perform the training/validation loop for a neural network model for medical image registration.

    Args:
        args (Namespace): Arguments parsed from command line or configuration file.
        model (nn.Module): Neural network model to be trained/evaluated.
        warp_layer (nn.Module): STN Warping layer used at the end of the model.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        lr_scheduler: Learning rate scheduler.
        dice_metric_before: Dice metric MONAI object for masks before transformation.
        dice_metric_after: Dice metric MONAI object for masks after transformation.
        epoch (int): Current epoch number.
        max_epochs (int): Maximum number of epochs for training.
        loader: Data loader for loading batches of training/validation data.
        weight_sim (float): Weight for similarity loss.
        weight_reg (float): Weight for regularization loss.
        weight_dice (float): Weight for Dice loss.
        metrics (dict): Dictionary to store evaluation metrics.
        best_eval_loss (float): Best evaluation loss achieved so far.
        best_eval_dice (float): Best evaluation Dice coefficient achieved so far.
        pth_best_loss (str): Path to save the best model based on total loss.
        pth_best_dice (str): Path to save the best model based on Dice coefficient.
        dir_save (str): Directory to save trained models and logs.
        device: Device (CPU/GPU) to perform computation on.
        do_save (bool, optional): Whether to save intermediate results and models. Defaults to True.
        wandb_log (bool, optional): Whether to log metrics to Weights & Biases. Defaults to True.
        num_moving (int, optional): Number of moving images. Defaults to 10.
        num_save_samples (int, optional): Number of samples to save. Defaults to 2.
        teacher_model (nn.Module, optional): Teacher model for distillation-based training. Defaults to None.

    Returns:
        nn.Module: Trained neural network model.
        dict: Updated metrics dictionary.
        float: Best evaluation loss achieved so far.
        str: Path to the best model based on total loss.
        float: Best evaluation Dice coefficient achieved so far.
        str: Path to the best model based on Dice coefficient.
    """
    t0_train = time.time()
    mode = "Training" if model.training else "Validating"
    dict_prefix = "train" if model.training else "val"
    epoch_loss, epoch_sim_loss, epoch_reg_loss, epoch_dice_loss, n_steps= 0, 0, 0, 0, 0
    epoch_agreement_loss = 0
    epoch_distillation_loss = 0      
    epoch_distillation_losses = [0 for i in range(num_moving)]
    epoch_sim_losses, epoch_reg_losses, epoch_dice_losses = [0 for i in range(num_moving)],[0 for i in range(num_moving)],[0 for i in range(num_moving)]
    # moving_seg, fixed_seg = None, None
    dices_before,dices_after = defaultdict(list),defaultdict(list)
    amplitudes = None
    for batch_data in tqdm.tqdm(loader, desc=f"{mode}..."):
        # if weight_dice >0 :
        fixed_image, fixed_seg = batch_data["fixed_image"].to(device).squeeze(-1), batch_data["fixed_seg"].to(device).squeeze(-1)
        moving_images = [batch_data[f"moving_image_{i}"].to(device).squeeze(-1) for i in range(num_moving)]
        moving_segs = [batch_data[f"moving_seg_{i}"].to(device).squeeze(-1) for i in range(num_moving)]
        if (args.time_encoding_dim is not None):
            amplitudes = batch_data["delta_amplitudes"].to(device)
        # delta_amplitude = batch_data["delta_amplitude"].to(device)
        n_steps += fixed_image.shape[0]
        # Forward pass and loss
        if mode == "Training":
            optimizer.zero_grad()
        if teacher_model is not None:
            flows_teacher, flows, moved_images, moved_segs = forward_n(fixed_image, moving_images, moving_segs, model, warp_layer, amplitudes,fixed_as_input=args.fixed_as_input,
                                                                        teacher_model=teacher_model,amplitudes_wtih_fixed=batch_data["delta_amplitudes_with_fixed"].to(device))
            
        else:
            # np.concatenate(([0],moving_amplitudes))
            flows, moved_images, moved_segs = forward_n(fixed_image, moving_images, moving_segs, model, warp_layer, amplitudes,fixed_as_input=args.fixed_as_input)
            flows_teacher = None
        (sim_losses, reg_losses, dice_losses,
            total_sim_loss,total_reg_loss, total_dice_loss,
            total_loss,total_agreement_loss,
            distillation_losses, total_distillation_loss) = loss_fun_simple_n(fixed_image, moved_images,fixed_seg,moved_segs,flows,
                                            weight_sim,weight_reg,weight_dice,
                                            device, agreement_weight=args.agreement_weight,
                                            flows_teacher=flows_teacher, weight_distillation = args.weight_distillation)
        if mode == "Training":
            # Optimise
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()

        epoch_loss += total_loss.item() * fixed_image.shape[0]
        epoch_sim_loss += total_sim_loss.item() * fixed_image.shape[0]
        epoch_reg_loss += total_reg_loss.item() * fixed_image.shape[0]
        epoch_dice_loss += total_dice_loss.item() * fixed_image.shape[0]
        epoch_agreement_loss += total_agreement_loss.item() * fixed_image.shape[0]

        epoch_distillation_loss += total_distillation_loss.item() * fixed_image.shape[0]
        for i in range(len(sim_losses)):
            epoch_sim_losses[i] += sim_losses[i] * fixed_image.shape[0]
            epoch_reg_losses[i] += reg_losses[i] * fixed_image.shape[0]
            epoch_dice_losses[i] += dice_losses[i] * fixed_image.shape[0]

            epoch_distillation_losses[i] += distillation_losses[i] * fixed_image.shape[0]

            
            # Append for dice:
            # if not (mode == "Training"):
            # Append items
            
            moved_seg = moved_segs[i].round()
            moving_seg = moving_segs[i].round()
            fixed_seg = fixed_seg.round()

            dice_metric_before(y_pred=moving_seg, y=fixed_seg)
            dice_metric_after(y_pred=moved_seg, y=fixed_seg)



            
            ####### NEWWW SHOULD BE DONE PER OUTPUT .........
            dice_before = dice_metric_before.aggregate().item()
            dice_metric_before.reset()
            dice_after = dice_metric_after.aggregate().item()
            dice_metric_after.reset()
            
            dices_before[f"{i}"].append(dice_before)
            dices_after[f"{i}"].append(dice_after)
            dices_before["all"].append(dice_before)
            dices_after["all"].append(dice_after)
    if not mode == "Training":
        # Scheduler step
        if args.scheduler == "Plateau":
            lr_scheduler.step(epoch_loss/n_steps)
        else:
            lr_scheduler.step()
    # Loss
    epoch_loss /= n_steps
    epoch_sim_loss /= n_steps
    epoch_reg_loss /= n_steps
    epoch_dice_loss /= n_steps
    epoch_agreement_loss /= n_steps
    epoch_distillation_loss /= n_steps
    # Conserve metrics:
    metrics[f"{dict_prefix}/loss"].append(epoch_loss)
    metrics[f"{dict_prefix}/sim_loss"].append(epoch_sim_loss)
    metrics[f"{dict_prefix}/reg_loss"].append(epoch_reg_loss)
    metrics[f"{dict_prefix}/dice_loss"].append(epoch_dice_loss)
    metrics[f"{dict_prefix}/agreement_loss"].append(epoch_agreement_loss)

    metrics[f"{dict_prefix}/distillation_loss"].append(epoch_distillation_loss)
    for i in range(len(sim_losses)):
        epoch_sim_losses[i] /= n_steps
        epoch_reg_losses[i] /= n_steps
        epoch_dice_losses[i] /= n_steps
        metrics[f"{dict_prefix}/sim_loss_{i}"].append(epoch_sim_losses[i])
        metrics[f"{dict_prefix}/reg_loss_{i}"].append(epoch_reg_losses[i])
        metrics[f"{dict_prefix}/dice_loss_{i}"].append(epoch_dice_losses[i])

        metrics[f"{dict_prefix}/distillation_loss_{i}"].append(epoch_distillation_losses[i])

    ##### Save sample Images #####
    if do_save :
        for i in range(num_save_samples):
            sample_fixed,sample_moving,sample_moved = fixed_image[0,...],moving_images[i][0,...],moved_images[i][0,...]
            fig,axes = plt.subplots(ncols=2, nrows=2, figsize=(20,20))
            axes[0][0].imshow(torch.abs(sample_fixed-sample_moving).squeeze().cpu().detach())
            axes[0][1].imshow(torch.abs(sample_fixed-sample_moved).squeeze().cpu().detach())
            axes[1][0].imshow(sample_moving.squeeze().cpu().detach())
            axes[1][1].imshow(sample_moved.squeeze().cpu().detach())
            axes[0][0].set_title(f"Diff before {torch.sum(torch.abs(sample_fixed-sample_moving)).squeeze().cpu().detach().numpy()}")
            axes[0][1].set_title(f"Diff after {torch.sum(torch.abs(sample_fixed-sample_moved)).squeeze().cpu().detach().numpy()} ")
            axes[1][0].set_title("Moving ")
            axes[1][1].set_title("Moved")
            plt.savefig(f"{dir_save}/sample_diff_images_{mode}_{i}.png")
            plt.close()
            # if weight_dice > 0 :
            sample_fixed,sample_moving,sample_moved = fixed_seg[0,...],moving_segs[i][0,...],moved_segs[i][0,...]
            labels_fixed = mask_to_labels(sample_fixed.detach().cpu())
            labels_moving = mask_to_labels(sample_moving.detach().cpu())
            labels_moved = mask_to_labels(sample_moved.detach().cpu())

            fig,axes = plt.subplots(ncols=2, nrows=2, figsize=(20,20))
            axes[0][0].imshow(torch.abs(labels_fixed-labels_moving).squeeze().cpu().detach())
            axes[0][1].imshow(torch.abs(labels_fixed-labels_moved).squeeze().cpu().detach())
            axes[1][0].imshow(labels_moving.squeeze().cpu().detach())
            axes[1][1].imshow(labels_moved.squeeze().cpu().detach())
            axes[0][0].set_title(f"Diff before {torch.sum(torch.abs(labels_fixed-labels_moving)).squeeze().cpu().numpy()}")
            axes[0][1].set_title(f"Diff after {torch.sum(torch.abs(labels_fixed-labels_moved)).squeeze().cpu().numpy()} ")
            axes[1][0].set_title("Moving labels")
            axes[1][1].set_title("Moved labels")
            plt.savefig(f"{dir_save}/sample_diff_segs_{mode}_{i}.png")
            plt.close()
    # Print Epoch info :
    print(f"{mode} Epoch {epoch + 1}/{max_epochs} : ")
    print(f"___________ TOTAL {mode} INFO ___________")
    print(f"\t loss: {epoch_loss:.3f}")
    detailed_info = f"\t  Sim loss: {epoch_sim_loss:.3f} , Reg loss: {epoch_reg_loss:.3f} ,  Dice loss ={epoch_dice_loss:.3f} "
    print(detailed_info)
    for i in range(len(sim_losses)):
        print(f"\t ______ Moving {i} ______")
        detailed_info_i = f"\t  Sim loss: {epoch_sim_losses[i]:.3f} , Reg loss: {epoch_reg_losses[i]:.3f} ,  Dice loss ={epoch_dice_losses[i]:.3f} "
        print(detailed_info_i)
    print(f"\t elapsed time: {time.time()-t0_train:.2f} sec.")
    print("__________________________________________________")


    for i in range(len(sim_losses)):
        # Mean the metrics and print and log
        mean_dice_before = np.mean(dices_before[f"{i}"])
        mean_dice_after = np.mean(dices_after[f"{i}"])
        print(f"\t ---- For {i} : dice_before: {mean_dice_before:.3f}, dice_after: {mean_dice_after:.3f}")
        metrics[f"{dict_prefix}/dice_{i}"].append(mean_dice_after)
        metrics[f"{dict_prefix}/dice_before_{i}"].append(mean_dice_before)
    mean_dice_before = np.mean(dices_before["all"])
    mean_dice_after = np.mean(dices_after["all"])
    print(f"\t ---- For mean : dice_before: {mean_dice_before:.3f}, dice_after: {mean_dice_after:.3f}")
    metrics[f"{dict_prefix}/dice"].append(mean_dice_after)
    metrics[f"{dict_prefix}/dice_before"].append(mean_dice_before)
    
    # Additional validation metrics and save model eventually..:
    if not (mode == "Training"):
        
        if total_loss.item() < best_eval_loss: 
            best_eval_loss = total_loss.item()
            # Save best model based on dice
            if pth_best_loss != "":
                os.remove(os.path.join(dir_save, pth_best_loss))
            pth_best_loss = f"best_total_loss_{epoch + 1}_{best_eval_loss:.3f}.pth"
            torch.save(model.state_dict(), os.path.join(dir_save, pth_best_loss))
            print(f"{epoch + 1} | Saving best TRE model: {pth_best_loss}")

        if mean_dice_after > best_eval_dice:
            print(f"Save change .. {mean_dice_after=} > {best_eval_dice=}")
            best_eval_dice = mean_dice_after
            print(f"After change .. {mean_dice_after=} > {best_eval_dice=}")
            if do_save:
                # Save best model based on Dice
                if pth_best_dice != "":
                    os.remove(os.path.join(dir_save, pth_best_dice))
                pth_best_dice = f"best_dice_{epoch + 1}_{best_eval_dice:.3f}.pth"
                torch.save(model.state_dict(), os.path.join(dir_save, pth_best_dice))
                print(f"{epoch + 1} | Saving best Dice model: {pth_best_dice}")
        if wandb_log:
                # metrics["lr"] = optimizer.param_groups[0]['lr']
                wandb.log({k: v[-1] for k, v in metrics.items()} | {"lr": optimizer.param_groups[0]['lr']})


    return model, metrics, best_eval_loss, pth_best_loss, best_eval_dice, pth_best_dice


def train(args,max_epochs, model, warp_layer, optimizer,lr_scheduler,
                    train_loader, val_loader, weight_sim,weight_reg,weight_dice,
                    dir_save, device,do_save=True, wandb_log=True):
    # INIT METRICS
    dice_metric_before = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_after = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    metrics = defaultdict(list)
    pth_best_dice, pth_best_loss = "", ""
    best_eval_dice, best_eval_loss = 0, 1e8
    for epoch in range(max_epochs):
        # Train :
        model.train()
        (model, metrics, _, _, _, _) = train_loop(args, model,warp_layer,optimizer,lr_scheduler,
               dice_metric_before,dice_metric_after,
               epoch, max_epochs,
               train_loader, weight_sim,weight_reg,weight_dice,
               metrics,
               1e8, 0, "", "", 
               dir_save,
               device, do_save=do_save, wandb_log=wandb_log)

        # Val :
        model.eval()
        with torch.no_grad():
            (model, metrics, 
            best_eval_loss, pth_best_loss,
            best_eval_dice, pth_best_dice) = train_loop(args, model,warp_layer,optimizer,lr_scheduler,
                                            dice_metric_before,dice_metric_after,
                                            epoch, max_epochs,
                                            val_loader, weight_sim,weight_reg,weight_dice,
                                            metrics,
                                            best_eval_loss, best_eval_dice, pth_best_loss, pth_best_dice,  
                                            dir_save,
                                            device, do_save=do_save, wandb_log=wandb_log)
            
    return model, metrics


def train_n_inputs(args,max_epochs, model, warp_layer, optimizer,lr_scheduler,
                    train_loader, val_loader, weight_sim,weight_reg,weight_dice,
                    dir_save, device,do_save=True, wandb_log=True,num_moving=10,num_save_samples=2,
                    teacher_model=None):
    # INIT METRICS
    dice_metric_before = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_after = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # Start torch training loop
    metrics = defaultdict(list)
    pth_best_dice, pth_best_loss = "", ""
    best_eval_dice, best_eval_loss = 0, 1e8
    for epoch in range(max_epochs):
        # Train :
        model.train()
        (model, metrics, 
        _, _,
        _, _) = train_loop2(args,model, warp_layer, optimizer,lr_scheduler,
                                            dice_metric_before,dice_metric_after,
                                            epoch, max_epochs,
                                            train_loader, weight_sim,weight_reg,weight_dice,
                                            metrics,
                                            1e8, 0, "", "", 
                                            dir_save,
                                            device, do_save=do_save, wandb_log=wandb_log,
                                            num_moving=num_moving,
                                            num_save_samples=num_save_samples,
                                            teacher_model=teacher_model)

        # Val :
        model.eval()
        with torch.no_grad():
            (model, metrics, 
            best_eval_loss, pth_best_loss,
            best_eval_dice, pth_best_dice) = train_loop2(args,model, warp_layer, optimizer,lr_scheduler,
                                                dice_metric_before,dice_metric_after,
                                                epoch, max_epochs,
                                                val_loader, weight_sim,weight_reg,weight_dice,
                                                metrics,
                                                best_eval_loss, best_eval_dice, pth_best_loss, pth_best_dice, 
                                                dir_save,
                                                device, do_save=do_save, wandb_log=wandb_log,
                                                num_moving=num_moving,
                                                num_save_samples=num_save_samples,
                                                teacher_model=teacher_model)
    return model, metrics