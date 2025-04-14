import numpy as np

import torch
from torch.utils.data import Dataset

    
class target_4DCT_dataset(Dataset):
    """ 4DCT construction dataset"""

    def __init__(self, dataset_image, dataset_image_no_preprocess, data_dict_indexes, fixed_as_input=True):
        self.dataset_image = dataset_image
        self.dataset_image_no_preprocess = dataset_image_no_preprocess
        self.data_dict_indexes = data_dict_indexes
        self.num_moving = ((len(data_dict_indexes[0])-1)//3)-1
        
        if not fixed_as_input:
            self.num_moving = (len(data_dict_indexes[0])-1)//3 #- 1
        # self.num_moving = (len(data_dict_indexes[0])//3)-1
        # if not fixed_as_input:
        #     self.num_moving = (len(data_dict_indexes[0])-1)//3 #- 1
        self.fixed_as_input = fixed_as_input

    def __len__(self):
        return len(self.data_dict_indexes)
    
    def _get_inputs(self,dict_indexes):
        curr_dict = {}
        if "moving_AcqNumber" in dict_indexes.keys():
            fixed_idx, moving_idx = dict_indexes["fixed_AcqNumber"],dict_indexes["moving_AcqNumber"]
            
            curr_dict["fixed_image"] = self.dataset_image[fixed_idx]["image"]
            
            curr_dict["moving_image"] = self.dataset_image[moving_idx]["image"]
            curr_dict["moving_image_no_preprocess"] = self.dataset_image_no_preprocess[moving_idx]["image"]
            # And Auxiliary info :
            curr_dict["fixed_amplitude"] = dict_indexes[f"fixed_amplitude"]
            curr_dict["moving_amplitude"] = dict_indexes[f"moving_amplitude"]
            curr_dict["fixed_phase"] = dict_indexes[f"fixed_phase"]
            curr_dict["moving_phase"] = dict_indexes[f"moving_phase"]
            curr_dict["meta"] = self.dataset_image[fixed_idx]["image_meta_dict"]
        elif (self.fixed_as_input):
            fixed_idx = dict_indexes["fixed_AcqNumber"]
            curr_dict["fixed_amplitude"] = dict_indexes[f"fixed_amplitude"]
            curr_dict["fixed_image"] = self.dataset_image[fixed_idx]["image"]
            curr_dict["fixed_phase"] = dict_indexes[f"fixed_phase"]
            curr_dict["meta"] = self.dataset_image[fixed_idx]["image_meta_dict"]
            # And all movings:
            moving_indexes = [dict_indexes[f"moving_AcqNumber_{i}"] for i in range(self.num_moving)]
            moving_amplitudes = [dict_indexes[f"moving_amplitude_{i}"] for i in range(self.num_moving)]
            moving_amplitudes_with_fixed =np.concatenate(([0], moving_amplitudes))

            # Construct moving dict:
            for i, idx in enumerate(moving_indexes):
                curr_dict[f"moving_image_{i}"] = self.dataset_image[idx]["image"]
                curr_dict[f"moving_image_no_preprocess_{i}"] = self.dataset_image_no_preprocess[idx]["image"]
                curr_dict[f"moving_phase_{i}"] = dict_indexes[f"moving_phase_{i}"]
            curr_dict["delta_amplitudes"] = torch.Tensor(moving_amplitudes).unsqueeze(1)
            curr_dict["delta_amplitudes_with_fixed"] = torch.Tensor(moving_amplitudes_with_fixed).unsqueeze(1)
            curr_dict["chosen_idx_for_result"] = torch.Tensor([dict_indexes["chosen_idx_for_result"]])
            
        else:
            # All movings:
            moving_indexes = [dict_indexes[f"moving_AcqNumber_{i}"] for i in range(self.num_moving)]
            moving_amplitudes = [dict_indexes[f"moving_amplitude_{i}"] for i in range(self.num_moving)]
            curr_dict["meta"] = self.dataset_image[moving_indexes[0]]["image_meta_dict"]
            # Construct moving dict:
            for i, idx in enumerate(moving_indexes):
                curr_dict[f"moving_image_{i}"] = self.dataset_image[idx]["image"]
                curr_dict[f"moving_image_no_preprocess_{i}"] = self.dataset_image_no_preprocess[idx]["image"]
                curr_dict[f"moving_phase_{i}"] = dict_indexes[f"moving_phase_{i}"]
            curr_dict["delta_amplitudes"] = torch.Tensor(moving_amplitudes).unsqueeze(1)
            curr_dict["chosen_idx_for_result"] = torch.Tensor([dict_indexes["chosen_idx_for_result"]])
        return curr_dict


    def __getitem__(self, idx):
        curr_dict_indexes = self.data_dict_indexes[idx]
        data_dict = self._get_inputs(curr_dict_indexes)
        return data_dict
    

class infer_missing_frame_dataset(Dataset):
    """ 4DCT construction dataset"""

    def __init__(self, dataset, data_dict_indexes, fixed_as_input=True):
        """
        Arguments:
            dataset: dataset constructed by function in src, contains all images and segmentations in Acquisition Number order
            data_dict_indexes : Actual dataset of tuples of 11 images (1 fixed, 10 moving) giving their aacquistion number. 
                                These are then loaded from the "dataset" object
        """
        self.dataset = dataset
        self.data_dict_indexes = data_dict_indexes
        self.num_moving = (len(data_dict_indexes[0])-1)//2
        self.fixed_as_input = fixed_as_input

    def __len__(self):
        return len(self.data_dict_indexes)

    def _get_inputs(self,dict_indexes):
        reference_idx = dict_indexes["reference_AcqNumber"]
        curr_dict = {}
        curr_dict["reference_image"] = self.dataset[reference_idx]["image"]
        curr_dict["reference_seg"] = self.dataset[reference_idx]["seg"]
        if self.fixed_as_input:
            fixed_idx, moving_idx = dict_indexes["fixed_AcqNumber"],dict_indexes["moving_AcqNumber"]
            curr_dict["fixed_image"] = self.dataset[fixed_idx]["image"]
            curr_dict["fixed_seg"] = self.dataset[fixed_idx]["seg"]

            curr_dict["moving_image"] = self.dataset[moving_idx]["image"]
            curr_dict["moving_seg"] = self.dataset[moving_idx]["seg"]


            curr_dict["fixed_amplitude"] = dict_indexes[f"fixed_amplitude"]
            curr_dict["moving_amplitude"] = dict_indexes[f"moving_amplitude"]
        else:
            moving_indexes = [dict_indexes[f"moving_AcqNumber_{i}"] for i in range(self.num_moving)]
            moving_amplitudes = [dict_indexes[f"moving_amplitude_{i}"] for i in range(self.num_moving)]
            
            for i, idx in enumerate(moving_indexes):
                curr_dict[f"moving_image_{i}"] = self.dataset[idx]["image"]
                curr_dict[f"moving_seg_{i}"] = self.dataset[idx]["seg"]
            curr_dict["delta_amplitudes"] = torch.Tensor(moving_amplitudes).unsqueeze(1)
        return curr_dict


    def __getitem__(self, idx):
        curr_dict_indexes = self.data_dict_indexes[idx]
        data_dict = self._get_inputs(curr_dict_indexes)
        return data_dict




class CT4D_data_2_normal_dicom_loader(Dataset):
    """ 4DCT construction dataset"""

    def __init__(self, dataset, data_dict_indexes):
        """
        Arguments:
            dataset: dataset constructed by function in src, contains all images and segmentations in Acquisition Number order
            data_dict_indexes : Actual dataset of tuples of 11 images (1 fixed, 10 moving) giving their aacquistion number. 
                                These are then loaded from the "dataset" object
        """
        self.dataset = dataset
        self.data_dict_indexes = data_dict_indexes
        self.num_moving = (len(data_dict_indexes[0])-1)//2

    def __len__(self):
        return len(self.data_dict_indexes)

    def _get_inputs(self,dict_indexes):
        
        fixed_idx = dict_indexes["fixed_AcqNumber"]
        moving_index = dict_indexes["moving_AcqNumber"]
            
        curr_dict = {}
        curr_dict["fixed_image"] = self.dataset[fixed_idx]["image"]
        curr_dict["fixed_seg"] = self.dataset[fixed_idx]["seg"]

        curr_dict[f"moving_image"] = self.dataset[moving_index]["image"]
        curr_dict[f"moving_seg"] = self.dataset[moving_index]["seg"]
        return curr_dict

    def __getitem__(self, idx):
        curr_dict_indexes = self.data_dict_indexes[idx]
        data_dict = self._get_inputs(curr_dict_indexes)
        return data_dict


class CT4D_data_11_dicom_loader(Dataset):
    """ 4DCT construction dataset"""

    def __init__(self, dataset, data_dict_indexes, fixed_as_input=True):
        """
        Arguments:
            dataset: dataset constructed by function in src, contains all images and segmentations in Acquisition Number order
            data_dict_indexes : Actual dataset of tuples of 11 images (1 fixed, 10 moving) giving their aacquistion number. 
                                These are then loaded from the "dataset" object
        """
        self.dataset = dataset
        self.data_dict_indexes = data_dict_indexes
        self.num_moving = (len(data_dict_indexes[0])-1)//2
        self.fixed_as_input = fixed_as_input

    def __len__(self):
        return len(self.data_dict_indexes)

    def _get_inputs(self,dict_indexes):
        # print(dict_indexes)
        # print("______")
        fixed_idx = dict_indexes["fixed_AcqNumber"]
        moving_indexes = [dict_indexes[f"moving_AcqNumber_{i}"] for i in range(self.num_moving)]
        moving_amplitudes = [dict_indexes[f"moving_amplitude_{i}"] for i in range(self.num_moving)]
        moving_amplitudes_with_fixed =np.concatenate(([0], [dict_indexes[f"moving_amplitude_{i}"] for i in range(self.num_moving)]))
        if self.fixed_as_input:
            moving_amplitudes = np.concatenate(([0],moving_amplitudes))
            
        curr_dict = {}
        # print(fixed_idx)
        # print(len(self.dataset))
        curr_dict["fixed_image"] = self.dataset[fixed_idx]["image"]
        curr_dict["fixed_seg"] = self.dataset[fixed_idx]["seg"]

        for i, idx in enumerate(moving_indexes):
            # print(i,self.dataset[idx]["image_meta_dict"]["filename_or_obj"])
            curr_dict[f"moving_image_{i}"] = self.dataset[idx]["image"]
            curr_dict[f"moving_seg_{i}"] = self.dataset[idx]["seg"]
        curr_dict["delta_amplitudes"] = torch.Tensor(moving_amplitudes).unsqueeze(1)

        curr_dict["delta_amplitudes_with_fixed"] = torch.Tensor(moving_amplitudes_with_fixed).unsqueeze(1)
        # print(curr_dict[f"moving_image_{i}"])
        return curr_dict


    def __getitem__(self, idx):
        curr_dict_indexes = self.data_dict_indexes[idx]
        data_dict = self._get_inputs(curr_dict_indexes)
        return data_dict


