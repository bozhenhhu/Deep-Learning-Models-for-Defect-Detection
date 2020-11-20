from albumentations import *
from imgaug import augmenters as iaa
import random


def get_transforms(phase='train', p=0.8):
    if phase == "train":
        list_transforms = []
        list_transforms.extend(
            [
                HorizontalFlip(), VerticalFlip(),
                # OneOf([
                #     IAAAdditiveGaussianNoise(),
                #     GaussNoise(),
                # ], p=0.1),
                # OneOf([
                #     MotionBlur(p=.2),
                #     MedianBlur(blur_limit=3, p=.1),
                #     Blur(blur_limit=3, p=.1),
                # ], p=0.1),
                # OneOf([
                #     GridDistortion(p=.1),
                #     IAAPiecewiseAffine(p=0.2),
                # ], p=0.3),
                # OneOf([
                #     CLAHE(clip_limit=2),
                #     IAASharpen(),
                #     IAAEmboss(),
                #     RandomBrightnessContrast(),
                # ], p=0.2),
                HueSaturationValue(p=0.3),
            ]
        )
        list_trfms = Compose(list_transforms, p=p)
        return list_trfms


#two ways to aug image: by get_transforms using albumentations library; by imgaug library
transforms = get_transforms(phase='train', p=0.8)
flip_horizontal = iaa.Fliplr(p=0.5)
flip_vertical = iaa.Flipud(p=0.5)
brightness = iaa.Add((-7, 7), per_channel=0.5)
contrast = iaa.ContrastNormalization((0.8, 1.6), per_channel=0.5)
gaussian_noise = iaa.AdditiveGaussianNoise(loc=0, scale=(0.03 * 255, 0.04 * 255), per_channel=0.5)


def additive_gaussian_noise(img, seed=None, std=(0, 0.4)):
    """ Add gaussian noise to the current image pixel-wise
    Parameters:
      std: the standard deviation of the filter will be between std[0] and std[0]+std[1]
    """
    if seed is None:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(seed)
    sigma = std[0] + random_state.rand() * std[1]
    gaussian_noise = random_state.randn(*img.shape) * sigma
    noisy_img = img + gaussian_noise
    # noisy_img = np.clip(noisy_img, 0, 1)
    return noisy_img


def random_brightness(img, seed=None, max_change=40):
    """ Change the brightness of img
    Parameters:
      max_change: max amount of brightness added/subtracted to the image
    """
    if seed is None:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(seed)
    brightness = random_state.randint(-20, max_change)
    brightness /= 100.
    new_img = img + brightness
    return np.clip(new_img, 0, 1)


def aug_image(my_image, my_label):
    image = my_image.copy()
    label = my_label.copy()
    if random.choice([0, 0, 1]):
        images = flip_horizontal.augment_images([image, label])
        image = images[0]
        label = images[1]
    if random.choice([0, 0, 1]):
        images = flip_vertical.augment_images([image, label])
        image = images[0]
        label = images[1]

    if random.choice([0, 0, 1]):
        image = brightness.augment_image(image)
    if random.choice([0, 0, 1]):
        image = contrast.augment_image(image)
    if random.choice([0, 0, 1]):
        image = gaussian_noise.augment_image(image)

    return image, label

