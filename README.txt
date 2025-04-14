________________________________________________________________
Supporting code for IMITATE: Image Registration with context for
4D-CT Artefact Correction.
________________________________________________________________



__________________________TRAINING______________________________
main_train.py orchestrates the training for the different models, both IMITATE and VoxelMorph style training.

Typically the results in the paper were obtained with: 

1 - for VoxelMorph style with 4 inputs :
python main_train.py --weight_sim 0.7 --weight_dice 0.3 --weight_reg 0.063 --agreement_weight 0.7 -n 2 -i 4 --no-full_res_training --fixed_as_input --csv_paths_train train_csvs --csv_paths_val train_val

2- for VoxelMorph style with 4 inputs + conidtional Unet with encoding amplitudes to dim 32 :
python main_train.py --weight_sim 0.7 --weight_dice 0.3 --weight_reg 0.063 --agreement_weight 0.7 -n 2 -i 4 -t 32 --no-full_res_training --fixed_as_input --csv_paths_train train_csvs --csv_paths_val train_val

3- for IMITATE with 4 inputs : 
python main_train.py --weight_sim 0.7 --weight_dice 0.3 --weight_reg 0.063 --agreement_weight 0.7 -n 2 -i 4 -t 32 --no-full_res_training --no-fixed_as_input --csv_paths_train train_csvs --csv_paths_val train_val


__________________________Inference______________________________

Use construct_4DCT.py with the saved model from training.

________________________________________________________________


Note:
    Conditional Attention Unet is defined in src/conditional_model.py
    A 4D-CT dataset containing amplitude respiration files (VXP) is required for the method. 


