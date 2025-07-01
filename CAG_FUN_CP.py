import os
import re
import cv2
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A

# 오름차순 정렬 함수
def natural_key(text) :
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', text)]


class ImageLoader(object) :
    def __init__(self, CAG_img_paths, CAG_msk_paths, FUNDUS_img_paths, FUNDUS_msk_paths) :
        self.CAG_img_paths = sorted(CAG_img_paths, key = natural_key)
        self.CAG_msk_paths = sorted(CAG_msk_paths, key = natural_key)
        self.FUNDUS_img_paths = sorted(FUNDUS_img_paths, key = natural_key)
        self.FUNDUS_msk_paths = sorted(FUNDUS_msk_paths, key = natural_key)
    
    def _validate_lengths(self) :
        if len(self.CAG_img_paths) != len(self.CAG_msk_paths) :
            warn = (f"[Warning] # of CAG image ≠ # of CAG mask")
            raise ValueError(warn)
        
        if len(self.FUNDUS_img_paths) != len(self.FUNDUS_msk_paths) :
            warn = (f"[Warning] # of FUNDUS image ≠ # of FUNDUS mask")
            raise ValueError(warn)
    
    def load(self) :
        print(f"# of CAG image : {len(self.CAG_img_paths)}")
        print(f"# of CAG mask : {len(self.CAG_msk_paths)}")
        print(f"# of FUNDUS image : {len(self.FUNDUS_img_paths)}")
        print(f"# of FUNDUS mask : {len(self.FUNDUS_msk_paths)}")
        
        self._validate_lengths()
        
        CAG_paths = [{"name" : self.CAG_img_paths[i].split('/')[-1], "image" : self.CAG_img_paths[i], "mask" : self.CAG_msk_paths[i]} for i in range(len(self.CAG_img_paths))]
        FUNDUS_paths = [{"name" : self.FUNDUS_img_paths[i].split('/')[-1], "image" : self.FUNDUS_img_paths[i], "mask" : self.FUNDUS_msk_paths[i]} for i in range(len(self.FUNDUS_img_paths))]
        
        return CAG_paths, FUNDUS_paths


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
    def __init__(self, FUNDUS_paths, FUNDUS_output_path) :
        self.FUNDUS_paths = FUNDUS_paths
        self.FUNDUS_output_path = FUNDUS_output_path
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
    
    def process(self, FUNDUS_path) :
        img_path = FUNDUS_path["image"]
        msk_path = FUNDUS_path["mask"]
        
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
    
    def FUNDUS_transform(self) :
        FUNDUS_outputs = list()
        
        os.makedirs(os.path.join(self.FUNDUS_output_path, "image"), exist_ok = True)
        os.makedirs(os.path.join(self.FUNDUS_output_path, "mask"), exist_ok = True)
        
        for FUNDUS in self.FUNDUS_paths :
            transform_img, transform_msk = self.process(FUNDUS)
            transform_img.save(f"{self.FUNDUS_output_path}/image/{FUNDUS['name']}.png")
            transform_msk.save(f"{self.FUNDUS_output_path}/mask/{FUNDUS['name']}.png")


class augment_FUNDUS_list(object) :
    def __init__(self, augment_FUNDUS_path) :
        self.augment_FUNDUS_path = augment_FUNDUS_path
        
        # os.makedirs(os.path.join(self.augment_FUNDUS_path, "image"), exist_ok = True)
        # os.makedirs(os.path.join(self.augment_FUNDUS_path, "mask"), exist_ok = True)
    
    def make_list(self) :
        augment_FUNDUS_img_paths = glob.glob(os.path.join(self.augment_FUNDUS_path, "image/*.png"))
        augment_FUNDUS_img_paths = sorted(augment_FUNDUS_img_paths, key = natural_key)
        
        augment_FUNDUS_msk_paths = glob.glob(os.path.join(self.augment_FUNDUS_path, "mask/*.png"))
        augment_FUNDUS_msk_paths = sorted(augment_FUNDUS_msk_paths, key = natural_key)
        
        augment_FUNDUS_paths = [{"name" : augment_FUNDUS_img_paths[i].split('/')[-1], "image" : augment_FUNDUS_img_paths[i], "mask" : augment_FUNDUS_msk_paths[i]} for i in range(len(augment_FUNDUS_img_paths))]
        
        return augment_FUNDUS_paths


class CAG_FUNDUS_Copy_Paste(object) :
    def __init__(self, CAG_paths, FUNDUS_paths, output_path) :
        self.CAG_paths = CAG_paths
        self.FUNDUS_paths = FUNDUS_paths
        self.output_path = output_path
        
        os.makedirs(os.path.join(self.output_path, "image"), exist_ok = True)
        os.makedirs(os.path.join(self.output_path, "mask"), exist_ok = True)
        
    def Copy_and_Paste(self, augment_num = None) :
        # num = 0
        if augment_num != None :
            for CAG in tqdm(self.CAG_paths, desc = "Copy and Paste ") :
                CAG_name = CAG["name"].split('.')[0]
                FUNDUS_sample = random.sample(self.FUNDUS_paths, augment_num)
                
                for FUNDUS in FUNDUS_sample :
                    FUNDUS_name = FUNDUS["name"].split('.')[0]
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
                    
                    blur_FUNDUS_img_ar = cv2.GaussianBlur(resize_FUNDUS_img_ar, ksize = (5, 5), sigmaX = 3)
                    blur_FUNDUS_msk_ar = cv2.GaussianBlur(resize_FUNDUS_msk_ar, ksize = (5, 5), sigmaX = 3)
                    
                    FUNDUS_img_PIL = Image.fromarray(blur_FUNDUS_img_ar + 20, mode = "L")
                    FUNDUS_msk_PIL = Image.fromarray(blur_FUNDUS_msk_ar, mode = "L")
                    FUNDUS_msk_PIL_2 = Image.fromarray(resize_FUNDUS_msk_ar, mode = "L")
                    
                    CAG_img.paste(FUNDUS_img_PIL, (paste_x, paste_y), mask = FUNDUS_msk_PIL)
                    CAG_msk.paste(255, (paste_x, paste_y), mask = FUNDUS_msk_PIL_2)
                    
                    FUNDUS_img_PIL.save(f"{self.output_path}/fundus_image/{FUNDUS_name}.png")
                    FUNDUS_msk_PIL_2.save(f"{self.output_path}/fundus_mask/{FUNDUS_name}.png")
                    CAG_img.save(f"{self.output_path}/augmented_image/{CAG_name}_{FUNDUS_name}.png")
                    CAG_msk.save(f"{self.output_path}/augmented_mask/{CAG_name}_{FUNDUS_name}.png")
                    
        else :
            for CAG in tqdm(self.CAG_paths, desc = "Copy and Paste ") :
                CAG_name = CAG["name"].split('.')[0]
                for FUNDUS in self.FUNDUS_paths :
                    FUNDUS_name = FUNDUS["name"].split('.')[0]
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
                    
                    blur_FUNDUS_img_ar = cv2.GaussianBlur(resize_FUNDUS_img_ar, ksize = (5, 5), sigmaX = 3)
                    blur_FUNDUS_msk_ar = cv2.GaussianBlur(resize_FUNDUS_msk_ar, ksize = (5, 5), sigmaX = 3)
                    
                    FUNDUS_img_PIL = Image.fromarray(blur_FUNDUS_img_ar + 20, mode = "L")
                    FUNDUS_msk_PIL = Image.fromarray(blur_FUNDUS_msk_ar, mode = "L")
                    FUNDUS_msk_PIL_2 = Image.fromarray(resize_FUNDUS_msk_ar, mode = "L")
                    
                    CAG_img.paste(FUNDUS_img_PIL, (paste_x, paste_y), mask = FUNDUS_msk_PIL)
                    CAG_msk.paste(255, (paste_x, paste_y), mask = FUNDUS_msk_PIL_2)
                    
                    FUNDUS_img_PIL.save(f"{self.output_path}/fundus_image/{FUNDUS_name}.png")
                    FUNDUS_msk_PIL_2.save(f"{self.output_path}/fundus_mask/{FUNDUS_name}.png")
                    # CAG_img.save(f"{self.output_path}/augmented_image/{num:05d}.png")
                    # CAG_msk.save(f"{self.output_path}/augmented_mask/{num:05d}.png")
                    # num += 1
                    
                    CAG_img.save(f"{self.output_path}/image/{CAG_name}_{FUNDUS_name}.png")
                    CAG_msk.save(f"{self.output_path}/mask/{CAG_name}_{FUNDUS_name}.png")

if __name__ == "__main__" :
    # CAG_path = "/path/to/CAG"
    # FUNDUS_path = "/path/to/FUNDUS"
    output_path = "/path/to/output"
    augment_FUNDUS_path = "/path/to/augmented_FUNDUS"
    
    # os.makedirs(os.path.join(CAG_path, "image"), exist_ok = True)
    # os.makedirs(os.path.join(CAG_path, "mask"), exist_ok = True)
    # os.makedirs(os.path.join(FUNDUS_path, "image"), exist_ok = True)
    # os.makedirs(os.path.join(FUNDUS_path, "mask"), exist_ok = True)
    
    # CAG_img_paths = glob.glob(os.path.join(CAG_path, "image/*.png"))
    # CAG_msk_paths = glob.glob(os.path.join(CAG_path, "mask/*.png"))
    # FUNDUS_img_paths = glob.glob(os.path.join(FUNDUS_path, "image/*.png"))
    # FUNDUS_msk_paths = glob.glob(os.path.join(FUNDUS_path, "mask/*.png"))
    
    CAG_img_paths = glob.glob(os.path.join("path/to", "image/*.png"))
    CAG_msk_paths = glob.glob(os.path.join("path/to", "mask/*.png"))
    FUNDUS_img_paths = glob.glob(os.path.join("path/to", "image/*.png"))
    FUNDUS_msk_paths = glob.glob(os.path.join("path/to", "mask/*.png"))
    
    CAG_paths, FUNDUS_paths = ImageLoader.load(CAG_img_paths, CAG_msk_paths, FUNDUS_img_paths, FUNDUS_msk_paths)
    # FUNDUS_outputs = FUNDUS_ImageProcess(FUNDUS_paths, augment_FUNDUS_path).FUNDUS_transform()
    FUNDUS_ImageProcess(FUNDUS_paths, augment_FUNDUS_path).FUNDUS_transform()
    augment_FUNDUS_paths = augment_FUNDUS_list(augment_FUNDUS_path).make_list()
    # augment_num = 1 : 1배수
    CAG_FUNDUS_Copy_Paste(CAG_paths, augment_FUNDUS_paths, output_path).Copy_and_Paste(augment_num = 1)
