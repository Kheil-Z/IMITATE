
import glob
import numpy as np
import os 
import csv
from monai import transforms as MTransforms
import torch

import nibabel as nib
from src.train_utils import forward, forward_n
from src.data_utils import  get_run,make_model, make_4DCT_target_dataset
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace


from monai import transforms as MTransforms
from monai.networks.blocks import Warp
import tqdm
import random

from monai.losses import LocalNormalizedCrossCorrelationLoss

import argparse


# Function to score all slices
def score_by_slice(image, correlation_func):
    res = []
    for i in range(1,image.shape[-1]):
        slice_before,slice_after = image[:,:,i-1],image[:,:,i]
        score = correlation_func(torch.Tensor(slice_before).unsqueeze(0).unsqueeze(0),
                                 torch.Tensor(slice_after).unsqueeze(0).unsqueeze(0)).item()
        res.append(score)
    return res
    
# Function to score a constructed volume as described in https://pubmed.ncbi.nlm.nih.gov/24892346/
def score_4DCT_paper(image, correlation_func):
    res = []
    norm_res = []
    for i in range(7,image.shape[-1]-1,8):
        slice_7 = torch.Tensor(image[:,:,i-1]).unsqueeze(0).unsqueeze(0)
        slice_8 = torch.Tensor(image[:,:,i]).unsqueeze(0).unsqueeze(0)
        slice_9 = torch.Tensor(image[:,:,i+1]).unsqueeze(0).unsqueeze(0)
        slice_10 = torch.Tensor(image[:,:,i+2]).unsqueeze(0).unsqueeze(0)
        
        score7_8 = correlation_func(slice_7,slice_8).item()
        score8_9 = correlation_func(slice_8,slice_9).item()
        score9_10 = correlation_func(slice_9,slice_10).item()
        score = 0.5*(score7_8 + score9_10) - score8_9
        norm_score = score8_9/(0.5*(score7_8 + score9_10))
        res.append(score)
        norm_res.append(norm_score)
    return res,norm_res

def make_4D_slice(args,target_amplitude, batch_data, model, warp_layer, fixed_as_input, device, num_inputs, num_moving=1):
    not_moved = 0
    if fixed_as_input and num_inputs==2:
        fixed_image = batch_data["fixed_image"].to(device).squeeze(-1)
        moving_image = batch_data["moving_image"].to(device).squeeze(-1)
        moving_image_no_preprocess = batch_data["moving_image_no_preprocess"].to(device).squeeze(-1)
        # Forward :
        flow, _, _ = forward(fixed_image, moving_image, torch.zeros_like(moving_image), model, warp_layer)
        # Adapt flow to what we ant:
        fixed_amplitude, moving_amplitude = batch_data["fixed_amplitude"].squeeze().numpy(),batch_data["moving_amplitude"].squeeze().numpy()
        factor = np.abs((moving_amplitude-target_amplitude)/(moving_amplitude-fixed_amplitude))
        flow /= factor 
        # Warp image and image with no preprocess:
        result_slice = warp_layer(moving_image, flow)
        result_slice_no_preprocess = warp_layer(moving_image_no_preprocess, flow)

        if (args.threshold>0):
            diff_amp_fixed = np.abs(fixed_amplitude-target_amplitude)
            diff_amp_moving = np.abs(moving_amplitude-target_amplitude)
            best_fit = min(diff_amp_moving,diff_amp_fixed)
            if best_fit <= args.threshold:
                not_moved = 1 
                if best_fit == diff_amp_fixed:
                    result_slice=fixed_image
                    result_slice_no_preprocess = fixed_image#None
                else:
                    result_slice=moving_image
                    result_slice_no_preprocess = moving_image#moving_image_no_preprocess
    else:
        moving_images = [batch_data[f"moving_image_{i}"].to(device).squeeze(-1) for i in range(num_moving)]
        fixed_image = torch.zeros_like(moving_images[0]).to(device)
        moving_segs = [torch.zeros_like(moving_images[i]).to(device) for i in range(num_moving)]
        amplitudes = batch_data["delta_amplitudes"].to(device)  
        choose_idx = int(batch_data["chosen_idx_for_result"].squeeze().numpy())#np.argmin(torch.abs(amplitudes).squeeze().cpu().numpy())
        if fixed_as_input :
            # find 
            fixed_amplitude = batch_data["fixed_amplitude"].squeeze().numpy()
            moving_amplitude = fixed_amplitude-amplitudes.squeeze()[choose_idx].cpu().numpy()

            amplitudes = batch_data["delta_amplitudes_with_fixed" ].to(device)  
            fixed_image = batch_data["fixed_image"].to(device).squeeze(-1)
            factor = np.abs((moving_amplitude-target_amplitude)/(moving_amplitude-fixed_amplitude))
            if factor < 1:
                factor = 1/factor
        # Forward pass :
        flows, _, _ = forward_n(fixed_image, moving_images, moving_segs, model, warp_layer, amplitudes,fixed_as_input=fixed_as_input)
        flow = flows[choose_idx]
        if fixed_as_input :
            flow /= factor 
        # Warp image and image with no preprocess:
        moving_image = moving_images[choose_idx]
        moving_image_no_preprocess = batch_data[f"moving_image_no_preprocess_{choose_idx}"].to(device).squeeze(-1) #TODO... 0 ???
        result_slice = warp_layer(moving_image, flow)
        result_slice_no_preprocess = warp_layer(moving_image_no_preprocess, flow)



        if (args.threshold>0):
            diff_amp_chosen = np.abs(amplitudes.squeeze().cpu()[choose_idx])
            diff_amp_other = np.abs(amplitudes.squeeze().cpu()[choose_idx+1])
            equivalent_index = choose_idx+1
            if fixed_as_input:
                diff_amp_chosen = np.abs( (fixed_amplitude-amplitudes.squeeze().cpu().numpy()[choose_idx+1])-target_amplitude)
                diff_amp_other = np.abs(fixed_amplitude-target_amplitude)
            if (not fixed_as_input) and (amplitudes.squeeze().cpu()[choose_idx]*amplitudes.squeeze().cpu()[choose_idx+1] >= 0):
                diff_amp_other = np.abs(amplitudes.squeeze().cpu()[choose_idx-1])
                equivalent_index = choose_idx-1
            
            best_fit = min(diff_amp_chosen,diff_amp_other)
            if best_fit <= args.threshold:
                not_moved = 1 
                if best_fit == diff_amp_chosen:
                    result_slice=moving_images[choose_idx]
                    result_slice_no_preprocess = moving_images[choose_idx]#None
                elif fixed_as_input:
                    result_slice=fixed_image
                    result_slice_no_preprocess = fixed_image
                else:
                    result_slice=moving_images[equivalent_index]
                    result_slice_no_preprocess = moving_images[equivalent_index]# None
    return result_slice.cpu(),result_slice_no_preprocess.cpu(),not_moved

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--patient_path')
    parser.add_argument('--target', type=float)
    parser.add_argument('--threshold',default=-1, type=float)
    parser.add_argument('--wandb_project', '-w')

    parser.add_argument('--save_nifty', action='store_true')
    parser.add_argument('--no-save_nifty', dest='save_nifty', action='store_false')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--no-eval', dest='eval', action='store_false')
    
    args = parser.parse_args()

    set_seed(42)
    assert (args.target>=0 and args.target<=1), "Problem!"
    device = "cuda:0"
    # Load model and model arguments used:
    run = get_run(args.model_path, path=args.wandb_project)
    model_args = SimpleNamespace(**run.config)
    if not hasattr(model_args, "detrend"):
        model_args.detrend  = False
    if not hasattr(model_args, "work_on_phase"):
        model_args.work_on_phase  = True
    model = make_model(model_args)
    # or :
    model_path = glob.glob(f"test_wandb/{args.model_path}/best_dice*.pth")[0]
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)  
    model = model.eval()

    warp_layer = Warp().to(device)

    ############### Params and main code:
    # device, optimizer, epoch and batch settings
    # Dataset options
    if model_args.full_res_training:
        target_res = [512, 512]
        spatial_size = [
            -1,
            -1,
            -1,
        ]  # for Resized transform, [-1, -1, -1] means no resizing, use this when training challenge model
    else:
        target_res = [256, 256, -1]
        spatial_size = target_res 

    train_transforms =  MTransforms.Compose([
                        MTransforms.LoadImaged(keys=["image"], reader="itkreader", image_only=False, ensure_channel_first=(True)),
                        MTransforms.Orientationd(keys=["image"], axcodes="LAS"), 
                        MTransforms.ThresholdIntensityd(
                                keys=["image"],
                                threshold=-1000.0, # bcz : https://research.tue.nl/files/168210888/Puneet_B..pdf
                                cval=-1000.0,
                                above=True,
                                ),
                        MTransforms.ThresholdIntensityd(
                            keys=["image",],
                            threshold=400.0, # bcz : https://research.tue.nl/files/168210888/Puneet_B..pdf
                            cval=400.0,
                            above=False,
                            ),
                        MTransforms.ScaleIntensityd(keys=["image"],minv=0.0,maxv=1.0),
                        MTransforms.Resized(
                            keys=["image"],
                            mode=("trilinear"),
                            align_corners=(True),
                            spatial_size=spatial_size),
                ])
    no_preprocess_transforms = MTransforms.Compose([
                        MTransforms.LoadImaged(keys=["image"], reader="itkreader", image_only=False, ensure_channel_first=(True,False)),
                        MTransforms.Orientationd(keys=["image"], axcodes="LAS"),
                        MTransforms.Resized(
                            keys=["image"],
                            mode="trilinear",
                            align_corners=True,
                            spatial_size=spatial_size),
                ]) 
    include_amplitudes = model_args.time_encoding_dim is not None
    cache_rate = 0
    csv_path = args.patient_path
    if not hasattr(model_args, "fixed_as_input"):
        model_args.fixed_as_input = True
        model_args.num_model_inputs = 2 

        
    eval_set,data_dict_indexes = make_4DCT_target_dataset(csv_path, train_transforms, no_preprocess_transforms,
                                                        args.target, num_sensors=8, cache_rate=cache_rate,
                                                        num_inputs=model_args.in_channel, fixed_as_input=model_args.fixed_as_input,
                                                        plot_name=f"results/{args.model_path}",
                                                        detrend=model_args.detrend,work_on_phase=model_args.work_on_phase)
    eval_loader = DataLoader(eval_set, batch_size=1,shuffle=False, num_workers=0)
    




    full_size=True
    stack_of_slices = np.zeros((512, 512, len(eval_set)))
    if not full_size:
        stack_of_slices = np.zeros((256, 256, len(eval_set)))
    i = len(eval_set)-1
    with torch.no_grad():
        num_not_moved = 0
        for batch_data in tqdm.tqdm(eval_loader):
            affine = batch_data["meta"]["affine"].squeeze().numpy()
            result_slice,result_slice_no_preprocess,not_moved = make_4D_slice(args, args.target, batch_data,  model, warp_layer, model_args.fixed_as_input, device, model_args.in_channel ,num_moving=model_args.num_model_inputs-1)#num_moving)
            num_not_moved += not_moved
            slice_img = result_slice.squeeze()
            
            if full_size:
                slice_img = MTransforms.Resize(mode="trilinear",align_corners=True,spatial_size=(512,512))(slice_img.cpu().unsqueeze(0)).squeeze()
            stack_of_slices[:,:,i] = np.fliplr(slice_img.numpy())
            i-=1


    if args.eval:
        sim_loss_func = LocalNormalizedCrossCorrelationLoss(spatial_dims=2, kernel_size=5)
        scores_by_slice = score_by_slice(stack_of_slices, sim_loss_func)
        complete_score,norm_complete_score = score_4DCT_paper(stack_of_slices, sim_loss_func)

        sim_loss_func = torch.nn.MSELoss()
        
        scores_by_slice_MSE = score_by_slice(stack_of_slices, sim_loss_func)
        complete_score_MSE,norm_complete_score_MSE = score_4DCT_paper(stack_of_slices, sim_loss_func)

        data = [[csv_path],
                [num_not_moved,args.target,args.threshold,args.model_path],
                scores_by_slice,
                complete_score,
                norm_complete_score,
                scores_by_slice_MSE,
                complete_score_MSE,
                norm_complete_score_MSE,
                [np.mean(scores_by_slice), np.mean(complete_score), np.mean(norm_complete_score),np.mean(scores_by_slice_MSE), np.mean(complete_score_MSE), np.mean(norm_complete_score_MSE)]]
        name = csv_path.split("/niftis_reconstructed/")[1].replace("/","").replace(".csv","")
        root = f"results/reconstruction_metrics_final_i_hope/{args.model_path}/" # model_args.detrend
        if not os.path.exists(root):
            os.makedirs(root)
        filename = f"{root}{args.model_path}_detrend={model_args.detrend}_{args.threshold}_{args.target}_name={name}.txt"
        print(filename)
        with open(filename, 'w') as f:
            csv.writer(f, delimiter=';').writerows(data)
        


    if args.save_nifty:
        if full_size:
            affine[2][2]= 2.5
        else:
            affine[2][2]= 2.5/2
        
        nib_img = nib.Nifti1Image(stack_of_slices, affine)
                
        ornt = np.array([[0, 1],
                    [1, -1],
                    [2, -1]])
        img_orient = nib_img.as_reoriented(ornt)
        nib.save(img_orient, f"results/{args.model_path}_{args.threshold}_{args.target}.nii.gz")
        print(f"For {args.model_path} and thresh = {args.threshold}, not moved: {num_not_moved}/{len(eval_set)}")