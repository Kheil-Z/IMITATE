import pandas as pd
import numpy as np


from scipy import signal
import torch
from monai.networks.nets import RegUNet, BasicUNet, UNETR, UNet, SegResNet, AttentionUnet
from src.data import target_4DCT_dataset , CT4D_data_11_dicom_loader, infer_missing_frame_dataset, CT4D_data_2_normal_dicom_loader
from src.conditional_model import AttentionUnet_with_time
from monai.data import Dataset, CacheDataset, PersistentDataset
from monai import transforms as MTransforms
import wandb
import matplotlib.pyplot as plt



#keep
def to_one_hot_from_label_dict(label_dict, mask):
    """ Transform a 3D segmentation mask to one hot format, using a label dictionary.
    Label dictionary can contain classes which will be mapped to the same one-hot label value.
    This is used to go from >50 classes to a more manageable channel size.

    Args:
        label_dict (dict[int:list[int]]): Dictionarry mapping labels.
                                        Keys are the channel position to use for one hot encoding.
                                        Values are the lists of classes to be mapped to that channel.
        mask (torch.Tensor): 3D segmentation mask, of shape[h,w,d]

    Returns:
        torch.Tensor: transformed 3D mask to one-hot labels using logic defined in dictionary.
                    [c,h,w,d] where c is the number of unique classses defined in the dict (it's length).
    """
    mask = mask.long()
    one_hot_seg = torch.zeros((len(label_dict),mask.shape[-3],mask.shape[-2],mask.shape[-1]))
    for label in label_dict.keys():
        curr_mask = torch.zeros(1, mask.shape[-3],mask.shape[-2],mask.shape[-1])
        for old_label in label_dict[label]:
            curr_mask += mask==old_label
        curr_mask = (curr_mask>0)
        one_hot_seg[label-1,:,:,:] = curr_mask.float()
    return one_hot_seg
#keep
def to_one_hot(mask,num_labels=8):
    """ Transforms a 3D segmentation mask to one hot format, using specified number of labels.
        For correct functionning, make sure mask contains num_labels unique labels, and that these are contiguous values.

    Args:
        mask (torch.Tensor): 3D segmentation mask, of shape[h,w,d]
        num_labels (int, optional): Number of unique classes to one-hot to. Defaults to 8.

    Returns:
        torch.Tensor: transformed 3D mask to one-hot labels. shape : [num_labels,h,w,d]
    """
    mask = mask.long()
    one_hot_seg = torch.zeros((num_labels,mask.shape[-3],mask.shape[-2],mask.shape[-1]))
    for label in range(1,9):
        curr_mask = mask==label
        curr_mask = (curr_mask>0)
        one_hot_seg[label-1,:,:,:] = curr_mask.float()
    return one_hot_seg

#keep
def to_one_hot_2d_from_label_dict(label_dict, mask):
    """ Identical to 'to_one_hot_from_label_dict'  but for 2D masks.

    Args:
        label_dict (dict[int:list[int]]): Dictionarry mapping labels.
                                        Keys are the channel position to use for one hot encoding.
                                        Values are the lists of classes to be mapped to that channel.
        mask (torch.Tensor): 2D segmentation mask, of shape[h,w]

    Returns:
        torch.Tensor: transformed 2D mask to one-hot labels using logic defined in dictionary.
                    [c,h,w] where c is the number of unique classses defined in the dict (it's length).
    """
    mask = mask.long()
    one_hot_seg = torch.zeros((len(label_dict),mask.shape[-2],mask.shape[-1]))
    for label in label_dict.keys():
        curr_mask = torch.zeros(1,mask.shape[-2],mask.shape[-1])
        for old_label in label_dict[label]:
            curr_mask += mask==old_label
        curr_mask = (curr_mask>0)
        one_hot_seg[label-1,:,:] = curr_mask.float()
    return one_hot_seg
#keep
def to_one_hot_2d(mask,num_labels=8):
    """Identical to 'to_one_hot'  but for 2D masks.

    Args:
        mask (torch.Tensor): 2D segmentation mask, of shape[h,w,d]
        num_labels (int, optional): Number of unique classes to one-hot to. Defaults to 8.

    Returns:
        torch.Tensor: transformed 2D mask to one-hot labels. shape : [num_labels,h,w]
    """
    mask = mask.long()
    one_hot_seg = torch.zeros((num_labels,mask.shape[-2],mask.shape[-1]))
    for label in range(1,9):
        curr_mask = mask==label
        curr_mask = (curr_mask>0)
        one_hot_seg[label-1,:,:] = curr_mask.float()
    return one_hot_seg


#keep
class One_Hot_Generald(MTransforms.MapTransform):
    """ General one-hot mask encoder Monai Transform.
        Uses specific number of labels, or label dictionaries specifier to encode segmentation masks to a usable one-hot format.
        In practice, this is requried for image registration as single-channel masks cannot be straightforwardly warped.
        Dictionnary based version.

    """
    def __init__(self, keys, origin_type="from_label", label_dict=None):
        """
        Args:
            keys (list[float]): List of keys to apply the transformation to
            origin_type (str, optional): Describes the type of one-hot to apply.
                                        If "from_label", will apply auto one-hot encoding with number of labels found.
                                        If anything else, will default to "from_dict", and expect a dictionary specifying the encoding scheme.
                                        Defaults to "from_label".
            label_dict (dict[int:list[int]], optional):Dictionarry mapping labels.
                                        Keys are the channel position to use for one hot encoding.
                                        Values are the lists of classes to be mapped to that channel.
                                        Defaults to None.
        """
        assert origin_type in ["from_dict", "from_label"], "Unrecognized one_hot origin type"
        super().__init__(keys)
        self.keys = keys
        self.label_dict = label_dict
        assert ((origin_type == "from_label")) or (label_dict is not None), "If 'from_dict', must specify label_dict argument"
        self.type = origin_type
        self.num_labels = 8 if label_dict is None else len(label_dict)

    def _to_one_hot_2d(self,item):
        """
            Calls approrpiate 2D one-hot encoder.
        """
        if self.type == "from_dict":
            mask = to_one_hot_2d_from_label_dict(self.label_dict, item)
        else:
            mask = to_one_hot_2d(item, num_labels=self.num_labels)
        return mask 

    def _to_one_hot(self,item):
        """
            Calls approrpiate 3D one-hot encoder.
        """
        if self.type == "from_dict":
            mask = to_one_hot_from_label_dict(self.label_dict, item)
        else:
            mask = to_one_hot(item, num_labels=self.num_labels)
        return mask 

    def __call__(self, data):
        """
            Apply the transform for specified keys of the input data dictionary.
        """
        for key in self.keys:
            item = data[key]
            assert (len(item.shape) == 3) or (len(item.shape) == 4), "Invalid shape"
            if len(item.shape) == 3 :
                mask = self._to_one_hot_2d(item)
            else:
                mask = self._to_one_hot(item)
            data[key] = mask
        return data


def get_run(name, path="wandb_project_name"):
    """
       Function to get model arguments used from wandb logs.

    Args:
        name (str): WandB model name.
        path (str, optional): path of the WandB project name.

    Returns:
        run (.): WandB run object.
    """
    api = wandb.Api()
    runs = api.runs(path=path)
    run = None
    for r in runs:
        if r.name == name :
            run = r
            break
    return run

def make_model(args):
    """Function to make registration model given arguments.

    Args:
        args (dict): Argument dictionary specifying the model type, input and output shapes. 

    Returns:
        Torch model: The required model.
    """
    # Model
    if args.model=="SegResNet":
        assert (args.time_encoding_dim is None), "Not Implemented"
        model = SegResNet(
            spatial_dims=2,
            in_channels=args.in_channel,
            out_channels=args.out_channel,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            dropout_prob=0.0)
    elif args.model=="UNet":
        assert (args.time_encoding_dim is None), "Not Implemented"
        model = UNet(
            spatial_dims=2, 
            in_channels=args.in_channel, 
            out_channels=args.out_channel,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2, 2),
            num_res_units=2)
    elif args.model=="attention":
        model = AttentionUnet_with_time(
            spatial_dims=2, 
            in_channels=args.in_channel, 
            out_channels=args.out_channel,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3, 
            dropout=0.0,
            time_encoding_dim=args.time_encoding_dim)
    return model



def make_pairwise_dataset(csv_paths, transforms, num_sensors=8, num_perumation=0,
                          cache_rate=0.,discard_prob=0.):
    """
    Create a "classic" pairwise registration dataset, with a moving and fixed (target image).
    This is done for all raw 4DCT paths specified in "csv_paths". 
    The function automatically generates approriate pairs, 
    i.e pair of images are acquired at the same table position, and by the same sensor number (only difference is due to breathing motion)
    Can be parametrised to return more or less permutations, hence have a bigger or more manageable dataset size.

    Args:
        csv_paths (list[str]): List of paths to .csv files describing a 4DCT dicom dataset.
        transforms (Monai transform object): Transforms to be applied to the CT dicoms and the nifti segmentations.
        num_sensors (int, optional): Size of sensor array used at acquisition. Defaults to 8.
        num_perumation (int, optional): By default, function will only pair contiguous images by order of acquisition.
                If a number > 0 is given, will generate additional number of randomly paired images at same location.
                Defaults to 0.
        cache_rate (float, optional): Specifies if using a CachedDataset mecanism or not. 
            If 0, no caching, otherwise will be used as cache rate,
            Defaults to 0..
        discard_prob (float, optional): Probability to not keep a given a pair in final dataset... 
            Used because dataset quickly becomes huge....
            Defaults to 0..

    Returns:
        Torch Dataset: Dataset required
    """
    # Difference with previous is : "offset" variable keeps track of how to much to offset access to a certain patient...
    # assert num_perumation < num_inputs, "Problem?"
    assert num_perumation < 11, "Problem?"
    data_dict_images = [] 
    data_dict_indexes = []
    offset = 0
    for csv_path in csv_paths:
        df =  pd.read_csv(csv_path)
        # Get list of table positions:
        table_positions = np.unique(df.AcquisitionNumber.to_numpy())
        # data_dict_indexes = []
        # For each table position:
        for table_position in table_positions:
            df_table_position = df[df.AcquisitionNumber == table_position]
            num_times = len(df_table_position)/num_sensors#len(df_table_position)//num_sensors
            assert num_times.is_integer(), "Problem..."# == 11, "Problem..."

            # Then construct data for each sensor:
            for sensor_number in range(num_sensors):
                current_sensor_idxs = df_table_position.InstanceNumber.to_numpy()[sensor_number::8]
                fixed_indexes = list(range(len(current_sensor_idxs)))
                for f in fixed_indexes:
                    remaining_idxs = [x for x in fixed_indexes if x!=f] # TODO should change.. 7/2
                    if f>0 and f<len(fixed_indexes):
                        curr_dict = {}
                        fixed_path_AcqNumber = current_sensor_idxs[f]
                        moving_path_AcqNumber = current_sensor_idxs[f-1]
                        # moving_idx = np.random.choice(len(remaining_idxs), size=num_inputs-1, replace=False, p=None)
                        if np.random.random() >= discard_prob:
                            curr_dict["fixed_AcqNumber"] = fixed_path_AcqNumber -1 + offset
                            curr_dict["moving_AcqNumber"] = moving_path_AcqNumber -1 + offset
                            data_dict_indexes.append(curr_dict)
                            # paths_used.append(current_paths)
                    elif num_perumation > 0:
                        moving_indexes = np.random.choice(remaining_idxs, size=num_perumation, replace=False, p=None)
                        for m in moving_indexes:
                            curr_dict = {}
                            fixed_path_AcqNumber = current_sensor_idxs[f]
                            moving_path_AcqNumber = current_sensor_idxs[m]
                            # moving_idx = np.random.choice(len(remaining_idxs), size=num_inputs-1, replace=False, p=None)
                            curr_dict["fixed_AcqNumber"] = fixed_path_AcqNumber -1 + offset
                            curr_dict["moving_AcqNumber"] = moving_path_AcqNumber -1 + offset
                            data_dict_indexes.append(curr_dict)
        # Load images :
        for path in df.FilePath.to_numpy():
            data_dict_images.append({"image": path, 
                            "seg": path.replace("/CT/", "/CT_segmentation/").replace(".dcm",".nii.gz")})
            offset += 1
        # dataset_images = CacheDataset(data=data_dict_images, transform=transforms)
    if cache_rate>0.:
        dataset_images = CacheDataset(data=data_dict_images, transform=transforms,cache_rate=cache_rate)#num_workers=None
    else:
        dataset_images = Dataset(data=data_dict_images, transform=transforms)
        
    complete_dataset = CT4D_data_2_normal_dicom_loader(dataset_images, data_dict_indexes)
    
    return complete_dataset


def make_n_dicom_dataset_optimal_caching_same_order_several_images(csv_paths, transforms, num_sensors=8, num_perumation=0,cache_rate=0., num_inputs=11,
                                        mode="closest-amp", fixed_as_input=True,
                                        detrend=False,work_on_phase=False):
    """_summary_

    Args:
        csv_paths (list[str]): List of paths to csvs describing the chosen patients' 4DCT. (paths to files, acquistion info etc..)
        transforms (monai Transforms): Transforms to be applied to images for model input
        num_sensors (int, optional): Number of sensors used in acquistion. Defaults to 8.
        cache_rate (float, optional): Cache rate to cache dataset. Defaults to 0..
        num_inputs (int, optional): Number of inputs required by model. Defaults to 11.
        mode (str, optional): Mode to construct the dataset, specifically fro moving images, can be:
            - "closest-amp": Moving images are the one with closest amplitude differences to the target.
            - Other, (implies "closest"): Moving images are the one acquired before/ after the target (chronological).
            Defaults to "closest-amp". Best is "closest" #TODO might delete... and default to "closest"
        num_perumation (int, optional): By default, dataset is constructed with only the first image acquired at a position.
                                        this behaviour is appropriate for a validation set, for training more matchings are desirable.
                                        This argument specifies if more or less pairs should be generated:
                                         = 0 : Only first pair.
                                         > 0 : num_perumation pairs.
                                         < 0 : All contiguous pairs.
                                         Defaults to 0.
        fixed_as_input (bool, optional): Specifies if model uses a fixed image at inference or not. Defaults to True.
        detrend (bool, optional): Wether to detrend amplitudes. Defaults to False.
        work_on_phase (bool, optional): Wether to work on phases rather than amplitudes. Defaults to False.

    Returns:
        Torch Dataset: Dataset required
    """
    # N.B: "offset" variable keeps track of how to much to offset access to a certain patient...
    # assert num_perumation < num_inputs, "Problem?"
    assert num_perumation < 11, "Problem?"
    assert (detrend==False) or (work_on_phase==False)
    data_dict_images = [] 
    data_dict_indexes = []
    offset = 0
    nopior = 0
    for csv_path in csv_paths:
        # print(csv_path)
        # Read 4DCT dicoms' detailed info :
        df =  pd.read_csv(csv_path)
        if detrend:
            df.amplitude = signal.detrend(df.amplitude)
        elif work_on_phase:
            df.amplitude = df.old_phase
        # Get list of table positions:
        table_positions = np.unique(df.AcquisitionNumber.to_numpy())
        # For each table position:
        for table_position in table_positions:
            df_table_position = df[df.AcquisitionNumber == table_position]
            num_times = len(df_table_position)/num_sensors
            assert num_times.is_integer(), "Problem..."

            # Then construct data for each sensor:
            for sensor_number in range(num_sensors):
                current_sensor_idxs = df_table_position.InstanceNumber.to_numpy()[sensor_number::8]
                current_amplitudes = df_table_position.amplitude.to_numpy()[sensor_number::8]
                # del_paths = df_table_position.FilePath.to_numpy()[sensor_number::8]
                # either all permutations, or none, or randomly in function of function arguments:
                if num_perumation == 0 :
                    fixed_indexes = [0]
                    num_items = 1
                elif num_perumation == -1 :
                    num_items = max(5,num_times-num_inputs)
                    fixed_indexes = list(range(len(current_sensor_idxs)))
                else:
                    num_items = min(num_perumation,num_times-num_inputs)
                    fixed_indexes = list(range(len(current_sensor_idxs)))

                if mode=="closest-amp": 
                    print("false")
                else:
                    fixed_indexes = list(range(len(current_sensor_idxs)))
                    num_before = (num_inputs-1)//2
                    num_after = (num_inputs-1)-num_before
                    fixed_indexes = fixed_indexes[num_before:-num_after]
                    if num_perumation == 0 :
                        fixed_indexes = np.random.choice(fixed_indexes, size=1, p=None)
                    if (num_perumation > 0) and ((num_perumation < len(fixed_indexes))) :
                        fixed_indexes = np.random.choice(fixed_indexes,replace=False, size=num_perumation, p=None)
                    for fixed_idx in fixed_indexes:
                        fixed_path_AcqNumber = current_sensor_idxs[fixed_idx]
                        fixed_amplitude = current_amplitudes[fixed_idx]
                        moving_indexes = [fixed_idx - i for i in range(num_before,0,-1)] + [fixed_idx + i for i in range(1,num_after+1)]
                        delta_amplitudes_with_fixed = [fixed_amplitude-a for a in current_amplitudes]
                        # Build data dict:
                        curr_dict = {}
                        curr_dict["fixed_AcqNumber"] = fixed_path_AcqNumber -1 + offset
                        for i,moving_index in enumerate(moving_indexes):
                            curr_moving_path_AcqNumber = current_sensor_idxs[moving_index]
                            curr_dict[f"moving_AcqNumber_{i}"] = curr_moving_path_AcqNumber -1 + offset
                            curr_dict[f"moving_amplitude_{i}"] = delta_amplitudes_with_fixed[moving_index]
                        data_dict_indexes.append(curr_dict)   
        # Conserver number of elements to construct appropriate model
        num_model_inputs = num_inputs
        # Load images :
        for path in df.FilePath.to_numpy():
            data_dict_images.append({"image": path, 
                "seg": path.replace("/CT/", "/CT_segmentation/").replace(".dcm",".nii.gz")})
            offset += 1
    # if cache_rate>0.:
    #     dataset_images = CacheDataset(data=data_dict_images, transform=transforms,cache_rate=cache_rate)
    # else:
    #     dataset_images = Dataset(data=data_dict_images, transform=transforms)
    dataset_images = PersistentDataset(data=data_dict_images, transform=transforms, cache_dir="temp_dir/")
    complete_dataset = CT4D_data_11_dicom_loader(dataset_images, data_dict_indexes, fixed_as_input=fixed_as_input)
    
    return complete_dataset, num_model_inputs



def make_4DCT_target_dataset(csv_path, transforms,transforms_no_preprocess, target_phase, num_sensors=8, cache_rate=0., num_inputs=11,
                            fixed_as_input=True, plot_name=None,
                            detrend=False,work_on_phase=False):
    """ 
    Create a dataset comprised of images and eventually amplitude differences 
    required by a model to reconstruct the 3DCT of a specified patient at a specified target amplitude or phase.
    In practice, for every slice location, the dataset will contain either:
        - Classic Approach: a fixed and a moving image acquired at the slice, and acquired before and after the target amplitude.
        - multi-input classic: like previous, but with several moving images around the target. + amplitude differences with fixed.
        - Ours : like previous, but with no fixed image, and ampltiude differences with target image.
    The function functions so that the three cases are warping the same image.
    Arguments mainly specify model requirements from the dataset. (Similar to "make_11_dicom_dataset_optimal_caching_same_order_several_images(.)")

    Args:
        csv_path (str): Path of the csv describing the patient's 4DCT. (paths to files, acquistion info etc..)
        transforms (monai Transforms): Transforms to be applied to images for model input
        transforms_no_preprocess (monai Transforms):  Additional transforms to load a copy of the images with other processing steps. 
            Typically used to reconstruct images with no pixel intensity processing, but reshaping and correct orientation.
        target_amplitude (float): The tagret amplitude to be reconstructed. Can be a phase, depends on "detrend" and "work_on_phase" args.
        num_sensors (int, optional): Number of sensors used in acquistion. Defaults to 8.
        cache_rate (float, optional): Cache rate to cache dataset. Defaults to 0..
        num_inputs (int, optional): Number of inputs required by model. Defaults to 11.
        fixed_as_input (bool, optional): Specifies if model uses a fixed image at inference or not. Defaults to True.
        plot_name (str | None, optional): If a string is given, will plot the amplitudes, chosen images and target amplitude and save plot.
            Defaults to None.
        detrend (bool, optional): Wether to detrend amplitudes. Defaults to False.
        work_on_phase (bool, optional): Wether to work on phases rather than amplitudes. Defaults to False.

    Returns:
        Torch Dataset: Dataset required
    """
    # Read 4DCT dicoms' detailed info :
    df =  pd.read_csv(csv_path)

    if detrend:
        df.amplitude = signal.detrend(df.amplitude)
    elif work_on_phase:
        df.amplitude = df.old_phase

    all_amps = df.amplitude.to_numpy()[0::8]
    chosen = np.zeros_like(all_amps)
    # Get list of table positions:
    table_positions = np.unique(df.AcquisitionNumber.to_numpy())
    data_dict_indexes = []
    targets = []
    pos = 0 
    for table_position in table_positions:
        df_table_position = df[df.AcquisitionNumber == table_position]
        num_times = len(df_table_position)/num_sensors
        assert num_times.is_integer(), "Problem..."# == 11, "Problem..."
        # Then construct data for each sensor:
        for sensor_number in range(num_sensors):
            current_sensor_idxs = df_table_position.InstanceNumber.to_numpy()[sensor_number::8]
            current_amplitudes = df_table_position.amplitude.to_numpy()[sensor_number::8]
            current_phases = df_table_position.old_phase_binned.to_numpy()[sensor_number::8]

            min_amp,max_amp = np.min(current_amplitudes), np.max(current_amplitudes)
            curr_target = ((1-target_phase)*min_amp) + (target_phase*max_amp)
            
            diff_amplitudes = np.array([a-curr_target for a in current_amplitudes])
            current_chosen = np.zeros_like(current_amplitudes)

            
            # Get list of indexes which will be used as the "missing" image target..
            possible_indexes = list(range(len(current_sensor_idxs)))#current_sensor_idxs
            curr_dict = {}
            if (fixed_as_input) and (num_inputs==2):
                # Get closest element (and make sure at least one acsuiqition before and after exists..):
                idx_min_difference = np.argmin(np.abs(diff_amplitudes[1:-2])) + 1
                # if phase_diff <= 0 :
                if (diff_amplitudes[idx_min_difference]*diff_amplitudes[idx_min_difference-1] >= 0):
                    moving_element = idx_min_difference
                    fixed_element = idx_min_difference + 1
                else:
                    moving_element = idx_min_difference -1
                    fixed_element = idx_min_difference

                # Get index from patient data:
                moving_idx = possible_indexes[moving_element]
                fixed_idx = possible_indexes[fixed_element]
                # Transform to path:
                fixed_path_AcqNumber = current_sensor_idxs[fixed_idx]
                moving_path_AcqNumber = current_sensor_idxs[moving_idx]
                # add to datat dict:
                curr_dict["fixed_AcqNumber"] = fixed_path_AcqNumber-1
                curr_dict["moving_AcqNumber"] = moving_path_AcqNumber-1
                curr_dict["fixed_amplitude"] = current_amplitudes[fixed_idx]
                curr_dict["moving_amplitude"] = current_amplitudes[moving_idx]
                curr_dict["fixed_phase"] = current_phases[fixed_idx]
                curr_dict["moving_phase"] = current_phases[moving_idx]
                if sensor_number ==0:
                    current_chosen[fixed_idx] = 0.5
                    current_chosen[moving_idx] = 1
            else:
                # 1)Start by finding closest element:
                # Need at least n/2 elements before and  after...:
                num_before = (num_inputs)//2
                num_after = (num_inputs)-num_before

                idx_min_difference = np.argmin(np.abs(diff_amplitudes[1:-2]))+1 #8
                if (idx_min_difference < num_before):
                    num_before = min(idx_min_difference,num_before)
                    num_after = (num_inputs)-num_before
                elif  (idx_min_difference >= len(diff_amplitudes)-num_after):
                    num_after = len(diff_amplitudes)-idx_min_difference
                    num_before = (num_inputs)-num_after#(num_inputs-1)-num_before
                fixed_idx = possible_indexes[idx_min_difference]

                # For with fixed:
                if fixed_as_input:
                    moving_indexes = [fixed_idx - i for i in range(num_before,0,-1)] + [fixed_idx + i for i in range(1,num_after)]#+ [fixed_idx + i for i in range(1,num_after+1)]
                    # add to datat dict:
                    fixed_amplitude = current_amplitudes[fixed_idx]
                    delta_amplitudes = [fixed_amplitude-a for a in current_amplitudes]
                    
                    fixed_path_AcqNumber = current_sensor_idxs[fixed_idx]
                    curr_dict["fixed_AcqNumber"] = fixed_path_AcqNumber -1
                    curr_dict["fixed_amplitude"] = fixed_amplitude
                    curr_dict["fixed_phase"] = current_phases[fixed_idx]

                    chosen_idx = num_before-1
                    for_plot = fixed_idx-1
                    if (diff_amplitudes[fixed_idx-1] * diff_amplitudes[fixed_idx] >=0) :
                        chosen_idx = num_before
                        for_plot = fixed_idx+1
                    # For plot:
                    if sensor_number ==0:
                        current_chosen[fixed_idx] = 0.5
                    
                # Fore no fixed:
                else:
                    mid_idx = possible_indexes[idx_min_difference]
                    moving_indexes = [mid_idx - i  for i in range(num_before,0,-1)] + [mid_idx + i for i in range(0,num_after)]
                    delta_amplitudes = [curr_target-a for a in current_amplitudes]
                    chosen_idx = num_before
                    for_plot = mid_idx
                for i,moving_index in enumerate(moving_indexes):
                    curr_moving_path_AcqNumber = current_sensor_idxs[moving_index]
                    curr_dict[f"moving_AcqNumber_{i}"] = curr_moving_path_AcqNumber -1
                    curr_dict[f"moving_amplitude_{i}"] = delta_amplitudes[moving_index]
                    curr_dict[f"moving_phase_{i}"] = current_phases[moving_index]
                    # For plot:
                    if sensor_number ==0:
                        current_chosen[moving_index] = 1
                curr_dict["chosen_idx_for_result"] = chosen_idx
                if sensor_number ==0:
                    current_chosen[for_plot] = 0.8
            data_dict_indexes.append(curr_dict)
            if sensor_number ==0:
                chosen[pos*len(current_chosen):(pos+1)*len(current_chosen)] = current_chosen
                pos+=1
        targets.append(curr_target)      
    # Load images :
    data_dict_images = []
    data_dict_images_only = []
    for path in df.FilePath.to_numpy():
        data_dict_images.append({"image": path})#, 
        data_dict_images_only.append({"image": path})
    if cache_rate>0.:
        dataset_image = CacheDataset(data=data_dict_images, transform=transforms,cache_rate=cache_rate)#num_workers=None
        dataset_image_no_preprocess = CacheDataset(data=data_dict_images_only, transform=transforms_no_preprocess,cache_rate=cache_rate)#num_workers=None
    else:
        dataset_image = Dataset(data=data_dict_images, transform=transforms)
        dataset_image_no_preprocess = Dataset(data=data_dict_images_only, transform=transforms_no_preprocess)
    
    complete_dataset = target_4DCT_dataset(dataset_image, dataset_image_no_preprocess, data_dict_indexes, fixed_as_input=fixed_as_input)
    if plot_name is not None:
        plt.figure(figsize=(15,8))
        plt.plot(all_amps)
        plt.scatter(list(range(len(all_amps))),all_amps,c=chosen)
        for i,y_val in enumerate(targets):
            plt.axhline(y=y_val,xmin = 0.04+(i/len(targets)), xmax = 0.04+((i+1)/len(targets)))
        for i in range(1,pos):
            plt.axvline(len(current_chosen)*i,color ='red')
        plt.savefig(f"{plot_name}.png")
        plt.close()
    return complete_dataset, data_dict_indexes          



def make_infer_missing_frame_dataset(csv_path, transforms, num_sensors=8, cache_rate=0., num_inputs=11,
                                    fixed_as_input=True,
                                    detrend=False,work_on_phase=False):
    """
        Porposed evaluation method: Infer an artificially missing image. 
        The function cosntructs the dataset for this. 
        To do so, it picks an iamge which we artificially "delete" from the dataset, 
        then gives the appropriate images required by a model to predict this "latent" image.
        Also returns the "missing" image and mask for evaluation.
        Arguments are similar to "make_4DCT_target_dataset(..)", and mainly specify model requirements.

    Args:
        csv_path (str): Path of the csv describing the patient's 4DCT. (paths to files, acquistion info etc..)
        transforms (monai Transforms): Transforms to be applied to images for model input
        num_sensors (int, optional): Number of sensors used in acquistion. Defaults to 8.
        cache_rate (float, optional): Cache rate to cache dataset. Defaults to 0..
        num_inputs (int, optional): Number of inputs required by model. Defaults to 11.
        fixed_as_input (bool, optional): Specifies if model uses a fixed image at inference or not. Defaults to True.
        detrend (bool, optional): Wether to detrend amplitudes. Defaults to False.
        work_on_phase (bool, optional): Wether to work on phases rather than amplitudes. Defaults to False.

    Returns:
        Torch Dataset: Dataset required
    """
    
    data_dict_images = [] 
    
    # Read 4DCT dicoms' detailed info :
    df =  pd.read_csv(csv_path)
    # Get list of table positions:
    table_positions = np.unique(df.AcquisitionNumber.to_numpy())
    data_dict_indexes = []
    # For each table position:
    if detrend:
        df.amplitude = signal.detrend(df.amplitude)
    elif work_on_phase:
        df.amplitude = df.old_phase
    for table_position in table_positions:
        df_table_position = df[df.AcquisitionNumber == table_position]
        num_times = len(df_table_position)/num_sensors
        assert num_times.is_integer(), "Problem..."

        # Then construct data for each sensor:
        for sensor_number in range(num_sensors):
            current_sensor_idxs = df_table_position.InstanceNumber.to_numpy()[sensor_number::8]
            current_amplitudes = df_table_position.amplitude.to_numpy()[sensor_number::8]
            
            # Get list of indexes which will be used as the "missing" image target..
            fixed_indexes = list(range(len(current_sensor_idxs)))#current_sensor_idxs

            num_before = (num_inputs)//2
            num_after = (num_inputs)-num_before
            fixed_indexes = fixed_indexes[num_before:-num_after]

            for missing_image_idx in fixed_indexes:
                missing_path_AcqNumber = current_sensor_idxs[missing_image_idx]
                missing_amplitude = current_amplitudes[missing_image_idx]
                curr_dict = {}
                curr_dict["reference_AcqNumber"] = missing_path_AcqNumber-1
                if fixed_as_input :
                    fixed_path_AcqNumber = current_sensor_idxs[missing_image_idx +1]
                    # fixed_amplitude = current_amplitudes[missing_image_idx +1]
                    moving_path_AcqNumber = current_sensor_idxs[missing_image_idx - 1]
                    # moving_amplitude = current_amplitudes[missing_image_idx - 1]

                    curr_dict["fixed_AcqNumber"] = fixed_path_AcqNumber-1
                    curr_dict["moving_AcqNumber"] = moving_path_AcqNumber-1
                    curr_dict["fixed_amplitude"] = current_amplitudes[missing_image_idx +1]
                    curr_dict["moving_amplitude"] = current_amplitudes[missing_image_idx - 1]
                else:
                    moving_indexes = [missing_image_idx - i for i in range(num_before,0,-1)] + [missing_image_idx + i for i in range(1,num_after+1)]
                    delta_amplitudes_with_missing = [missing_amplitude-a for a in current_amplitudes]
                    # Build data dict:
                    for i,moving_index in enumerate(moving_indexes):
                        curr_moving_path_AcqNumber = current_sensor_idxs[moving_index]
                        curr_dict[f"moving_AcqNumber_{i}"] = curr_moving_path_AcqNumber -1 
                        curr_dict[f"moving_amplitude_{i}"] = delta_amplitudes_with_missing[moving_index]       
                data_dict_indexes.append(curr_dict)
    # Load images :
    for path in df.FilePath.to_numpy():
        data_dict_images.append({"image": path, 
                        "seg": path.replace("/CT/", "/CT_segmentation/").replace(".dcm",".nii.gz")})
    if cache_rate>0.:
        dataset_images = CacheDataset(data=data_dict_images, transform=transforms,cache_rate=cache_rate)
    else:
        dataset_images = Dataset(data=data_dict_images, transform=transforms)  
    
    complete_dataset = infer_missing_frame_dataset(dataset_images, data_dict_indexes, fixed_as_input=fixed_as_input)

    return complete_dataset, data_dict_indexes
