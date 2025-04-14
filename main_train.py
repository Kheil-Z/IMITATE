
import glob
import numpy as np
import os 
import pandas as pd

from monai import transforms as MTransforms
import torch

from src.train_utils import train_n_inputs
from src.data_utils import  get_run,make_model, One_Hot_Generald, make_n_dicom_dataset_optimal_caching_same_order_several_images
import torch
import random
from torch.utils.data import DataLoader
from types import SimpleNamespace


from monai import transforms as MTransforms

from monai.networks.blocks import Warp
import wandb
import argparse

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
    parser.add_argument('--batch_size', '-b', default=64, type=int)
    parser.add_argument('--learning_rate', '-l', default=1e-3, type=float)
    parser.add_argument('--weight_decay', '-w', default=0, type=float)
    parser.add_argument('--optimizer', default="Adam")
    parser.add_argument('--scheduler', default="CosineAnnealing")
    parser.add_argument('--max_epochs', '-e', default=100, type=int)

    parser.add_argument('--weight_sim', '-s', default=0.8, type=float)
    parser.add_argument('--weight_reg', '-r', default=0.2*0.2, type=float)
    parser.add_argument('--weight_dice', '-d', default=0.8, type=float)

    parser.add_argument('--agreement_weight', default=0.0, type=float)

    
    parser.add_argument('--weight_distillation', default=0.0, type=float)
    parser.add_argument('--teacher_model_name', default="None")

    parser.add_argument('--num_perumation_train', '-n', default=5, type=int)
    parser.add_argument('--num_model_inputs', '-i', default=11, type=int)
    parser.add_argument('--model', '-m', default="attention")
    parser.add_argument('--time_encoding_dim', '-t', default=None, type=int)

    parser.add_argument('--data_mode', default="ordered")

    parser.add_argument("--cache_rate", default=0.4, type=float)
    
    
    # Using a fixed image, or training using INTERPOLATE
    parser.add_argument('--fixed_as_input', action='store_true')
    parser.add_argument('--no-fixed_as_input', dest='fixed_as_input', action='store_false')
    
    # Train in full resolution or half-res
    parser.add_argument('--full_res_training', action='store_true')
    parser.add_argument('--no-full_res_training', dest='full_res_training', action='store_false')

    # Log metrics to Weights and Biases
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--no-log_wandb', dest='log_wandb', action='store_false')

    # Detrend amplitudes, or work on phase data instead of amplitudes. Or none.
    parser.add_argument("--detrend",default="False")
    parser.add_argument("--work_on_phase",default="False")

    # List of csv file paths containing the information on the 4DCT of a patient.
    parser.add_argument("--csv_paths_train")
    parser.add_argument("--csv_paths_val")
    
    args = parser.parse_args()

    args.detrend = args.detrend == "True"
    args.work_on_phase = args.work_on_phase == "True"

    print(args.fixed_as_input)
    
    set_seed(42)
    ############### Params and main code:
    # device, optimizer, epoch and batch settings
    device = "cuda:0"
    batch_size = args.batch_size
    lr = args.learning_rate 
    weight_decay = 1e-5
    max_epochs = args.max_epochs

    # loss weights (set to zero to disable loss term)
    weight_sim = args.weight_sim
    weight_reg = args.weight_reg
    weight_dice = args.weight_dice

    #  Write model plot some useful figs?
    do_save = True

    # Dataset options
    if args.full_res_training:
        target_res = [512, 512]
        spatial_size = [
            -1,
            -1,
            -1,
        ]  # for Resized transform, [-1, -1, -1] means no resizing, use this when training challenge model
    else:
        target_res = [256, 256, -1]
        spatial_size = target_res 

    # Preprocessing Operations
    train_transforms =  MTransforms.Compose([
                        MTransforms.LoadImaged(keys=["image", "seg"], reader="itkreader", image_only=False, ensure_channel_first=(True,False)),
                        MTransforms.Orientationd(keys=["image"], axcodes="LAS"), # TODO was fixed
                        MTransforms.Orientationd(keys=["seg"], axcodes="RA"), # TODO was fixed
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
                            keys=["image", "seg"],
                            mode=("trilinear", "nearest"),
                            align_corners=(True, None),
                            spatial_size=spatial_size),
                        One_Hot_Generald(["seg"], origin_type="from_label", label_dict=None)#, allow_missing_keys=True)
                ])
    include_amplitudes = args.time_encoding_dim is not None
    cache_rate = args.cache_rate#0.4#8
    
    # Create Datasets
    train_set, found_num_model_inputs = make_n_dicom_dataset_optimal_caching_same_order_several_images(args.csv_paths_train, train_transforms, num_perumation=args.num_perumation_train,
                                                                               cache_rate=cache_rate, num_inputs=args.num_model_inputs,
                                                                               fixed_as_input=args.fixed_as_input,
                                                                               mode=args.data_mode,
                                                                               detrend=args.detrend,work_on_phase=args.work_on_phase)
    
                                                                   
    val_set, found_num_model_inputs = make_n_dicom_dataset_optimal_caching_same_order_several_images(args.csv_paths_val, train_transforms, num_perumation=0,
                                                                             cache_rate=cache_rate, num_inputs=args.num_model_inputs,
                                                                             fixed_as_input=args.fixed_as_input,
                                                                             mode=args.data_mode,
                                                                             detrend=args.detrend,work_on_phase=args.work_on_phase)

    print(len(train_set))
    print(len(val_set))
    
    assert args.num_model_inputs == found_num_model_inputs, "Sanity check failed...."
    num_moving = found_num_model_inputs-1

    # Make loaders :
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size,shuffle=False, num_workers=0)
    
    # Make model with chosen parameters :
    normal_reg = True
    args.in_channel = args.num_model_inputs
    args.out_channel = (args.num_model_inputs-1)*2
    if not args.fixed_as_input:
        args.in_channel = args.num_model_inputs -1

    model = make_model(args)
    model = model.to(device)  
    # Initialise weightless Warper layer
    warp_layer = Warp().to(device)
    
    teacher_model = None 
    if (args.teacher_model_name is not None) and (args.weight_distillation >0):
        run = get_run(args.teacher_model_name)
        print(f"Teacher model used config :")
        print(f"\t {run.config}")
        teacher_model = make_model(SimpleNamespace(**run.config))

        model_path = glob.glob(f"test_wandb/{args.teacher_model_name}/best_total_loss*.pth")[0]
        teacher_model.load_state_dict(torch.load(model_path))
        teacher_model = teacher_model.to(device)  
        teacher_model = teacher_model.eval()

    
    # Optimizer
    if args.optimizer == "Adam" :
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    else:
        args.optimizer = "SGD"
        optimizer =torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    if args.scheduler == "CosineAnnealing":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        args.scheduler == "Plateau"
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)#10) # TODO:

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.num_params = num_params
    if args.log_wandb :
        run = wandb.init(project="project_name",config=args)
        dir_save = f"test/{run.name}/"
    else:
        dir_save = f"test/"

    if do_save and not os.path.exists(dir_save):
        os.makedirs(dir_save)


    model, metrics = train_n_inputs(args,max_epochs, 
                                    model, warp_layer, 
                                    optimizer,lr_scheduler,
                                    train_loader, val_loader,
                                    weight_sim,weight_reg,weight_dice,
                                    dir_save, device,
                                    do_save=do_save, 
                                    wandb_log=args.log_wandb, num_moving=num_moving,num_save_samples=min(2,args.num_model_inputs-1),
                                    teacher_model=teacher_model)