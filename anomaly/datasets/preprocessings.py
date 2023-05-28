import cv2
import numpy as np
import math

from PIL import Image, ImageStat
from torchvision import transforms, utils


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEFAULT_IMG_SIZE = 224


class DefaultPreprocessing:
    def __init__(self, img_size) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN,
                                 std=STD)])
    
    def __call__(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img



class BlurAllKernelFivePreprocessing(DefaultPreprocessing):
    def __init__(self, img_size):
        super().__init__(img_size)
        self.transform = Compose([
            transforms.Resize(self.img_size),
            transforms.GaussianBlur(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN,
                                 std=STD)])


class BlurAllKernelNinePreprocessing(DefaultPreprocessing):
    def __init__(self, img_size) -> None:
        super().__init__(img_size)
        self.transform = Compose([
            transforms.Resize(self.img_size),
            transforms.GaussianBlur(9),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN,
                                 std=STD)])



class BlurUnblurredBase(DefaultPreprocessing):
    def __init__(self, img_size) -> None:
        super().__init__(img_size)
        self.kernel_size = None
    
    def __call__(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        value = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        if value > 50:
            img = cv2.blur(img, (self.kernel_size, self.kernel_size))

        img = Image.fromarray(img)
        img = self.transform(img)
        return img


class BlurUnblurredKernelFivePreprocessing(BlurUnblurredBase):
    def __init__(self, img_size) -> None:
        super(BlurUnblurredKernelFivePreprocessing, self).__init__(img_size)
        self.kernel_size = 5


class BlurUnblurredKernelNinePreprocessing(BlurUnblurredBase):
    def __init__(self, img_size) -> None:
        super(BlurUnblurredKernelNinePreprocessing, self).__init__(img_size)
        self.kernel_size = 9


class AdaptiveMeanThresholdPreprocessing(DefaultPreprocessing):
    def __init__(self, img_size) -> None:
        super().__init__(img_size)

    def __call__(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.medianBlur(img, 5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(gray ,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        img_th = np.zeros_like(img)
        img_th[:,:,0] = th
        img_th[:,:,1] = th
        img_th[:,:,2] = th

        img = Image.fromarray(img)
        img = self.transform(img)
        return img


class TripleThresholdPreprocessing(DefaultPreprocessing):
    def __init__(self, img_size) -> None:
        super().__init__(img_size)

    def __call__(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.medianBlur(img, 5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[:, :, 0] = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        img[:, :, 1] = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        _, img[:,:,2] = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        img = Image.fromarray(img)
        img = self.transform(img)
        return img


class SharpenAllPreprocessing(DefaultPreprocessing):
    def __init__(self, img_size) -> None:
        super().__init__(img_size)

    def __call__(self, img_path):
        img = Image.open(img_path).convert('RGB')

        sharpen = np.array([[0, -1, 0], [-1, 4.7, -1], [0, -1, 0]])
        img = np.array(img)
        img = cv2.filter2D(img, -1, sharpen) 

        img = Image.fromarray(img)
        img = self.transform(img)
        return img


class SharpenBlurredOnlyPreprocessing(DefaultPreprocessing):
    def __init__(self, img_size) -> None:
        super().__init__(img_size)

    def __call__(self, img_path):
        img = Image.open(img_path).convert('RGB')

        sharpen = np.array([[0, -1, 0], [-1, 4.7, -1], [0, -1, 0]])
        img = np.array(img)
        value = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        if value <= 50:
            img = cv2.filter2D(img, -1, sharpen) 

        img = Image.fromarray(img)
        img = self.transform(img)
        return img


class AddBrightnessPreprocessing(DefaultPreprocessing):
    def __init__(self, img_size) -> None:
        super().__init__(img_size)
        self.brightness_threshold = 20
        self.transform = Compose([
            py_vision.Resize(self.img_size),
            py_vision.ToTensor()])

    @staticmethod
    def get_brightness(im):
        im = im.convert('L')
        stat = ImageStat.Stat(im)
        return stat.mean[0]
    
    def __call__(self, img_path):
        img = Image.open(img_path).convert('RGB')
        brightness = AddBrightnessPreprocessing.get_brightness(img)

        if brightness < self.brightness_threshold:
            out = np.array(img)
            # compute slope and intercept  
            con = 20 
            diffcon = (100 - con)
            if diffcon <= 0.1: con=99.9

            arg = math.pi * (((con * con) / 20000) + (3 * con / 200)) / 4
            slope = 1 + (math.sin(arg) / math.cos(arg))
            if slope < 0: slope=0

            pivot = (100 - self.brightness_threshold) / 200
            intcpbri = self.brightness_threshold / 100
            intcpcon = pivot * (1 - slope)
            intercept = (intcpbri + intcpcon)

            # apply slope and intercept
            img = out/255.0
            out = slope * out + intercept
            out[out>1] = 1
            out[out<0] = 0

            out = 255.0 * out
            out = out.astype(np.uint8)

            img = Image.fromarray(out)

        img = self.transform(img)[0]
        return img


PREPROCESSINGS = {
    'default': DefaultPreprocessing,
    'blur_all_kernel_5': BlurAllKernelFivePreprocessing,
    'blur_all_kernel_9': BlurAllKernelNinePreprocessing,
    'blur_unblurred_only_kernel_5': BlurUnblurredKernelFivePreprocessing,
    'blur_unblurred_only_kernel_9': BlurUnblurredKernelNinePreprocessing,
    'adaptive_mean_threshold': AdaptiveMeanThresholdPreprocessing,
    'triple_threshold': TripleThresholdPreprocessing,
    'sharpen_all': SharpenAllPreprocessing,
    'sharpen_blurred_only': SharpenBlurredOnlyPreprocessing,
    'add_brightness_preprocessing': AddBrightnessPreprocessing
}