import glob
import gzip
import nibabel as nib
import numpy as np
import os
from PIL import Image
import pydicom
import shutil


def pngImgTo3dArray(patient_path: str) -> np.ndarray:
    """
    Converts a series of 2D PNG CT images from a specified directory into a 3D NumPy array.
    Note that PNG images in the specified directory are expected to be located in "ct" subfolder.

    Args:
        patient_path (str): The path to the directory containing the patient's CT scan images.

    Yields:
        np.ndarray: A 2D image slice, which can then be combined along the first dimension
                    by the caller into a complete 3D array.
    """
    image_paths = sorted(glob.glob(f"{patient_path}/ct/*.png"))
    for i in range(len(image_paths)):
        image = Image.open(image_paths[i])
        image = np.array(image)
        image = image[np.newaxis, :, :]
        yield image


def pngLblTo3dArray(patient_path: str) -> np.ndarray:
    """
    Converts a series of 2D PNG label images from a specified directory into a 3D NumPy array.
    Note that PNG images in the specified directory are expected to be located in "mask" subfolder.

    Args:
        patient_path (str): The path to the directory containing the patient's label images.

    Yields:
        np.ndarray: A 2D label image slice, which can then be combined along the first dimension
                    by the caller into a complete 3D array.
    """
    mask_paths = sorted(glob.glob(f"{patient_path}/mask/*.png"))
    for i in range(len(mask_paths)):
        mask = Image.open(mask_paths[i])
        mask = np.array(mask)
        mask = mask[np.newaxis, :, :]
        yield mask


def windowing(image: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    """
    Applies windowing to a medical image to enhance contrast by mapping the pixel values to a specified range.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        window_center (int): The center of the window.
        window_width (int): The width of the window.

    Returns:
        np.ndarray: The windowed image as a NumPy array with pixel values scaled to 0-255.
    """
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    windowed_image = np.clip(image, min_value, max_value).astype(np.float32)
    windowed_image = (windowed_image - min_value) / (max_value - min_value) * 255
    return windowed_image.astype(np.uint8)


def unzip_gz_files(folder_path: str) -> None:
    """
    Unzips all .gz files in the specified directory, extracting them to .nii files
    in the same directory. The original .gz files are not deleted.

    Args:
        folder_path (str): The path to the directory containing the .gz files.

    Returns:
        None
    """
    gzfiles = [gzfile for gzfile in os.listdir(folder_path) if ".gz" in gzfile]
    for gzfile in gzfiles:
        with gzip.open(f"{folder_path}/{gzfile}", "rb") as f_in:
            with open(f"{folder_path}/{gzfile.split('.')[0]}.nii", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":

    os.makedirs("npz_files", exist_ok=True)

    # Collect the #### dataset (#### studies)

    patients_paths = sorted(glob.glob("../3d_part_segmentation/training_data/data2d/*"))
    for id, pat_path in zip(range(1, len(patients_paths) + 1), patients_paths):
        img = np.vstack([slc for slc in pngImgTo3dArray(pat_path)])
        img = np.transpose(img, axes=(1, 2, 0))
        lbl = np.vstack([slc for slc in pngLblTo3dArray(pat_path)])
        lbl = np.transpose(lbl, axes=(1, 2, 0))
        new_lbl = np.zeros_like(lbl)
        new_lbl[lbl == 4] = 1
        new_lbl[lbl == 3] = 2
        new_lbl[lbl == 2] = 3
        new_lbl[lbl == 1] = 4
        new_name = f"pancreas_{id:04d}"
        # print(new_name, "image", img.shape, img.dtype, "label", new_lbl.shape, new_lbl.dtype)
        np.savez_compressed(f"npz_files/{new_name}", image=img, mask=new_lbl)


    # Collect the decathlon dataset (281 studies)

    cur_id = len(os.listdir("npz_files")) + 1
    folder_path = "../decathlon_dataset/ds_transformed"
    tensors = sorted(os.listdir(folder_path))
    for id, tensor_name in zip(range(cur_id, len(tensors) + cur_id), tensors):
        tensor_path = f"{folder_path}/{tensor_name}"
        with np.load(tensor_path) as data:
            img = data["image"]
            lbl = data["mask"].astype(np.uint8)
        new_lbl = np.zeros_like(lbl)
        new_lbl[lbl == 3] = 1
        new_lbl[lbl == 4] = 2
        new_lbl[lbl == 5] = 3
        new_lbl[lbl == 6] = 4
        new_name = f"pancreas_{id:04d}"
        # print(new_name, "image", img.shape, img.dtype, "label", new_lbl.shape, new_lbl.dtype)
        np.savez_compressed(f"npz_files/{new_name}", image=img, mask=new_lbl)


    # Collect the FLARE dataset (292 studies)

    image_folder = "../FLARE_dataset/vol"
    label_folder = "../FLARE_dataset/seg_2"
    dicom_folder = "../FLARE_dataset/dicom"
    # unzip_gz_files(image_folder)   # comment out if not needed
    # unzip_gz_files(label_folder)   # comment out if not needed

    old_ids = [i for i in range(1, 293)]
    old_ids.remove(85)
    cur_id = len(os.listdir("npz_files")) + 1
    for old_id, new_id in zip(old_ids, range(cur_id, len(os.listdir(image_folder)) + cur_id)):

        img_name = f"FLARE23_{old_id:04d}.nii"
        lbl_name = f"Segmentation_{old_id:04d}.nii"
        dcm_name = f"23_{old_id:04d}_0000.dcm"
        # print("Processing", img_name)

        img = nib.load(f"{image_folder}/{img_name}")
        lbl = nib.load(f"{label_folder}/{lbl_name}")
        change_x = img.header["srow_x"][0] < 0
        change_y = img.header["srow_y"][1] < 0
        change_z = img.header["srow_z"][2] < 0
        img_data = img.get_fdata(dtype=np.float32)
        lbl_data = lbl.get_fdata()

        if change_x:
            img_data = np.flip(img_data, axis=0)
            lbl_data = np.flip(lbl_data, axis=0)
        if change_y:
            img_data = np.flip(img_data, axis=1)
            lbl_data = np.flip(lbl_data, axis=1)
        if change_z:
            img_data = np.flip(img_data, axis=2)
            lbl_data = np.flip(lbl_data, axis=2)

        dicom_file = pydicom.dcmread(f"{dicom_folder}/{dcm_name}")
        window_center = dicom_file.WindowCenter
        window_width = dicom_file.WindowWidth
        img_data = windowing(img_data, window_center=window_center, window_width=window_width)

        img_tr = np.transpose(img_data, axes=(2, 0, 1))
        img_tr = np.array([np.transpose(sl[::-1, ::-1]) for sl in img_tr[::-1]])
        img_tr = np.transpose(img_tr, axes=(1, 2, 0)).astype(np.uint8)
        lbl_tr = np.transpose(lbl_data, axes=(2, 0, 1))
        lbl_tr = np.array([np.transpose(sl[::-1, ::-1]) for sl in lbl_tr[::-1]])
        lbl_tr = np.transpose(lbl_tr, axes=(1, 2, 0)).astype(np.uint8)

        new_name = f"pancreas_{new_id:04d}"
        np.savez_compressed(f"npz_files/{new_name}", image=img_tr, mask=lbl_tr)
