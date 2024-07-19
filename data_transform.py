import gzip
import nibabel as nib
import numpy as np
import os
import pandas as pd
import pydicom
import shutil


def unzip_masks(mask_folder_path: str) -> None:
    """
    Unzips all .gz mask files in the specified directory, extracting them to .nii files
    in the same directory. The original .gz files are not deleted.

    Args:
        mask_folder_path (str): The path to the directory containing the .gz mask files.

    Returns:
        None
    """
    mask_gzfiles = [gzfile for gzfile in os.listdir(mask_folder_path) if ".gz" in gzfile]
    for gzfile in mask_gzfiles:
        with gzip.open(f"{mask_folder_path}/{gzfile}", "rb") as f_in:
            with open(f"{mask_folder_path}/{gzfile.split('.')[0]}.nii", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


def construct3dImage(pan_paths: list[str]) -> np.ndarray:
    """
    Constructs a 3D NumPy array from a list of DICOM file paths.

    Args:
        pan_paths (list[str]): A list of paths to the DICOM files.

    Yields:
        np.ndarray: A 2D image slice from each DICOM file, which can then be combined
                    along the first dimension by the caller into a complete 3D array.
    """
    for pth in pan_paths:
        dicom_file = pydicom.dcmread(f"dicom_all/{pth}")
        dicom_array = dicom_file.pixel_array.astype(float)
        yield dicom_array


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
    windowed_image = np.clip(image, min_value, max_value)
    windowed_image = (windowed_image - min_value) / (max_value - min_value) * 255
    return windowed_image


def transform_dicom(image3d: np.ndarray, window_center: int = 67, window_width: int = 25) -> np.ndarray:
    """
    Applies normalization, windowing, and reorientation to a 3D image array.

    Args:
        image3d (np.ndarray): The input 3D image array.
        window_center (int, optional): Center of the windowing range. Default is 67.
        window_width (int, optional): Width of the windowing range. Default is 25.

    Returns:
        np.ndarray: The transformed 3D image array.
    """
    normalized_image = (image3d - np.min(image3d)) / (np.max(image3d) - np.min(image3d))
    scaled_image = (normalized_image * 255).astype(np.uint8)

    windowed_image = windowing(scaled_image, window_center, window_width)

    img_tr = np.array([sl[::-1, ::-1] for sl in windowed_image[::-1]])
    img_tr = np.transpose(img_tr, axes=(1, 2, 0))
    return img_tr.astype(np.uint8)


def transform_mask(nii_image: np.memmap) -> np.ndarray:
    """
    Transforms a 3D numpy array by reorienting slices and removing artifact values.

    Args:
        nii_image (np.memmap): 3D numpy array to be transformed.

    Returns:
        np.ndarray: Transformed 3D numpy array with specified values removed.
   """
    mask_tr = np.transpose(nii_image, axes=(2, 0, 1))
    mask_tr = np.array([np.transpose(sl[::-1, ::-1]) for sl in mask_tr[::-1]])
    mask_tr = np.transpose(mask_tr, axes=(1, 2, 0))
    # remove all 1 and 2 values
    mask_tr[(mask_tr == 1) | (mask_tr == 2)] = 0
    return mask_tr


if __name__ == "__main__":

    # Save dataset in a npz format for a custom Unet model

    os.makedirs("ds_transformed", exist_ok=True)
    pan_folders = sorted(os.listdir("dicom_all"))
    window_parameters = pd.read_csv("decathlon_window_parameters.csv")

    mask_folder_path = "label_nii"
    # unzip_masks(mask_folder_path)   # comment out if not needed
    mask_files = sorted(niifile for niifile in os.listdir(mask_folder_path) if ".gz" not in niifile)

    for npatient, mask in zip(pan_folders, mask_files):

        assert npatient == mask.split(".")[0], "Названия изображения и маски должны совпадать."

        window_center = window_parameters[window_parameters["study"] == npatient]["window_center"].iloc[0]
        window_width = window_parameters[window_parameters["study"] == npatient]["window_width"].iloc[0]

        pan_paths = sorted([f"{npatient}/{img}" for img in os.listdir(f"dicom_all/{npatient}")])
        pan_3d = np.array([img for img in construct3dImage(pan_paths)])
        img_trf = transform_dicom(pan_3d, window_center=window_center, window_width=window_width)

        mask_path = f"label_nii/{mask}"
        mask = nib.load(mask_path).get_fdata()
        msk_trf = transform_mask(mask)

        np.savez_compressed(f"ds_transformed/{npatient}", image=img_trf, mask=msk_trf)
