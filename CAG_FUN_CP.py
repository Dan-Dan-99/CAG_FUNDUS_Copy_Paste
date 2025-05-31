import os
import re
import cv2
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import albumentations as A

# 자연 정렬 함수
def natural_key(text) :
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', text)]


class ImageLoader(object) :
    def __init__(self, CAG_img_path : list[str], CAG_msk_path : list[str], FUNDUS_img_path : list[str], FUNDUS_msk_path : list[str]) :
        self.CAG_img_path = sorted(CAG_img_path, key = natural_key)
        self.CAG_msk_path = sorted(CAG_msk_path, key = natural_key)
        self.FUNDUS_img_path = sorted(FUNDUS_img_path, key = natural_key)
        self.FUNDUS_msk_path = sorted(FUNDUS_msk_path, key = natural_key)
    
    def _validate_lengths(self) :
        if len(self.CAG_img_path) != len(self.CAG_msk_path) :
            warn = (f"[Warning] # of CAG image ≠ # of CAG mask")
            raise ValueError(warn)
        
        if len(self.FUNDUS_img_path) != len(self.FUNDUS_msk_path) :
            warn = (f"[Warning] # of FUNDUS image ≠ # of FUNDUS mask")
            raise ValueError(warn)
    
    def load(self) :
        print(f"# of CAG image : {len(self.CAG_img_path)}")
        print(f"# of CAG mask : {len(self.CAG_msk_path)}")
        print(f"# of FUNDUS image : {len(self.FUNDUS_img_path)}")
        print(f"# of FUNDUS mask : {len(self.FUNDUS_msk_path)}")
        
        self._validate_lengths()
        
        CAG_path = [{"image" : self.CAG_img_path[i], "mask" : self.CAG_msk_path[i]} for i in range(len(self.CAG_img_path))]
        FUNDUS_path = [{"image" : self.FUNDUS_img_path[i], "mask" : self.FUNDUS_msk_path[i]} for i in range(len(self.FUNDUS_img_path))]
        
        return CAG_path, FUNDUS_path


class CustomErode(A.DualTransform) :
    def __init__(self, kernel_size = None, shape = None, iterations = 1, always_apply = False, p = 0.5) :
        super().__init__(always_apply, p)
        """
        shape = 0 : 8-way
        shape = 1 : 4-way
        shape = 2 : ellipse
        """
        if shape == 0 :
            shape = cv2.MORPH_RECT
        elif shape == 1 :
            shape = cv2.MORPH_CROSS
        elif shape == 2 :
            shape = cv2.MORPH_ELLIPSE
            
        self.kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
        self.iterations = iterations

    def apply(self, image, **params) :
        return cv2.erode(image, self.kernel, iterations = self.iterations)
    
    def apply_to_mask(self, mask, **params) :
        return cv2.erode(mask, self.kernel, iterations = self.iterations)

class CustomDilate(A.DualTransform) :
    def __init__(self, kernel_size = None, shape = None, iterations = 1, always_apply = False, p = 0.5) :
        super().__init__(always_apply, p)
        """
        shape = 0 : 8-way
        shape = 1 : 4-way
        shape = 2 : ellipse
        """
        if shape == 0 :
            shape = cv2.MORPH_RECT
        elif shape == 1 :
            shape = cv2.MORPH_CROSS
        elif shape == 2 :
            shape = cv2.MORPH_ELLIPSE
            
        self.kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
        self.iterations = iterations

    def apply(self, image, **params) :
        return cv2.dilate(image, self.kernel, iterations = self.iterations)
    
    def apply_to_mask(self, mask, **params) :
        return cv2.dilate(mask, self.kernel, iterations = self.iterations)

class FUNDUS_ImageProcess(object) :
    def __init__(self) :
        self.FUNDUS_transforms = A.Compose(
            [
                A.CLAHE(clip_limit = 8.0, tile_grid_size = (8, 8), p = 1),
                CustomErode(kernel_size = 3, shape = 0, iterations = 3, p = 1),
                CustomDilate(kernel_size = 9, shape = 0, iterations = 1, p = 1),
                A.GaussianBlur(blur_limit = (11, 11), sigma_limit = (40, 40), p = 1),
                A.HorizontalFlip(p = 0.5),
                A.VerticalFlip(p = 0.5),
                A.ShiftScaleRotate(shift_limit = 0.1, scale_limit = 0.2, rotate_limit = 20, p = 0.5)
            ],
            additional_targets = {"mask" : "mask"}
        )
    
    def process(self, FUNDUS_item : dict) -> tuple[Image.Image, Image.Image] :
        img_path = FUNDUS_item["image"]
        msk_path = FUNDUS_item["mask"]
        
        img = Image.open(img_path).convert("L")
        img_ar = np.array(img)
        
        msk = Image.open(msk_path).convert("L")
        msk_ar = (np.array(msk) == 255).astype(np.uint8)
        
        transform = self.FUNDUS_transforms(image = img_ar, mask = msk_ar)
        transform_img = transform["image"]
        transform_mak = transform["mask"]
        
        alpha = np.where(transform_mak > 0, 255, 0).astype(np.uint8)
        
        segment_img = np.where(transform_mak, transform_img, 0).astype(np.uint8)
        threshold_img = np.where(segment_img >= 120, 80, segment_img).astype(np.uint8)
        
        final_img = np.dstack([threshold_img, alpha])
        final_msk = np.dstack([transform_mak, alpha])
        
        final_img_PIL = Image.fromarray(final_img, mode = "LA")
        final_msk_PIL = Image.fromarray(final_msk, mode = "LA")
        
        return final_img_PIL, final_msk_PIL
    
    def FUNDUS_transform(self, FUNDUS_items : list[dict]) -> list[dict[str, Image.Image]] :
        FUNDUS_output = list()
        
        for FUNDUS in FUNDUS_items :
            transform_img, transform_msk = self.process(FUNDUS)
            FUNDUS_output.append({"image" : transform_img, "mask" : transform_msk})
        
        return FUNDUS_output


class CAG_FUNDUS_Copy_Paste(object) :
    def __init__(self, CAG_path : list[dict], FUNDUS_path : list[dict], output_path : str) :
        self.CAG_path = CAG_path
        self.FUNDUS_path = FUNDUS_path
        self.output_path = output_path
        
        os.makedirs(os.path.join(self.output_path, "augmented_image"), exist_ok = True)
        os.makedirs(os.path.join(self.output_path, "augmented_mask"), exist_ok = True)
        
    def Copy_and_Paste(self) :
        num = 0
        
        for CAG in tqdm(self.CAG_path, "Copy and Paste - ") :
            for FUNDUS in self.FUNDUS_path :
                CAG_img = Image.open(CAG["image"]).convert("L")
                CAG_img_ar = np.array(CAG_img)
                
                CAG_msk = Image.open(CAG["mask"]).convert("L")
                
                FUNDUS_img = Image.open(FUNDUS["image"]).convert("L")
                FUNDUS_img_ar = np.array(FUNDUS_img)
                
                FUNDUS_msk = Image.open(FUNDUS["mask"]).convert("L")
                FUNDUS_msk_ar = np.array(FUNDUS_msk)
                
                CAG_x = CAG_img_ar.shape[0]
                CAG_y = CAG_img_ar.shape[1]
                
                if CAG_x > CAG_y :
                    CAG_size = CAG_y
                elif CAG_x < CAG_y :
                    CAG_size = CAG_x
                else :
                    CAG_size = CAG_x
                
                FUNDUS_x = FUNDUS_img_ar.shape[0]
                FUNDUS_y = FUNDUS_img_ar.shape[1]
                
                if FUNDUS_x > FUNDUS_y :
                    FUNDUS_size = FUNDUS_y
                elif FUNDUS_x < FUNDUS_y :
                    FUNDUS_size = FUNDUS_x
                else :
                    FUNDUS_size = FUNDUS_x
                
                if CAG_size >= FUNDUS_size :
                    interpolate = cv2.INTER_CUBIC
                else :
                    interpolate = cv2.INTER_AREA
                
                scale_x = random.uniform(0.4, 0.6)
                scale_y = random.uniform(0.4, 0.6)
                
                img_size_x = int(CAG_size * scale_x)
                img_size_y = int(CAG_size * scale_y)
                
                if img_size_x >= img_size_y :
                    FUNDUS_max_size = img_size_x
                else :
                    FUNDUS_max_size = img_size_y
                
                paste_x = random.randint(20, CAG_size - FUNDUS_max_size - 20)
                paste_y = random.randint(20, CAG_size - FUNDUS_max_size - 20)
                
                resize_FUNDUS_img_ar = cv2.resize(FUNDUS_img_ar, (img_size_x, img_size_y), interpolation = interpolate)
                resize_FUNDUS_msk_ar = cv2.resize(FUNDUS_msk_ar, (img_size_x, img_size_y), interpolation = interpolate)
                _, resize_FUNDUS_msk_ar = cv2.threshold(resize_FUNDUS_msk_ar, 0, 255, type = cv2.THRESH_BINARY)

                blur_FUNDUS_img_ar = cv2.GaussianBlur(resize_FUNDUS_img_ar, kernel = (11, 11), sigmaX = 20)
                blur_FUNDUS_msk_ar = cv2.GaussianBlur(resize_FUNDUS_msk_ar, kernel = (11, 11), sigmaX = 20)
                
                FUNDUS_img_PIL = Image.fromarray(blur_FUNDUS_img_ar, mode = "L")
                FUNDUS_msk_PIL = Image.fromarray(blur_FUNDUS_msk_ar, mode = "L")
                
                CAG_img.paste(FUNDUS_img_PIL, (paste_x, paste_y), mask = blur_FUNDUS_msk_ar)
                CAG_msk.paste(FUNDUS_msk_PIL, (paste_x, paste_y), mask = blur_FUNDUS_msk_ar)
                
                CAG_img.save(f"{self.output_path}/augmented_image/{num:05d}.png")
                CAG_msk.save(f"{self.output_path}/augmented_mask/{num:05d}.png")
                
                num += 1
    
        print(f"Generate Complete !\nCAG {len(self.CAG_path)} * FUNDUS {len(self.FUNDUS_path)} = Total {len(self.CAG_path) * len(self.FUNDUS_path)}")