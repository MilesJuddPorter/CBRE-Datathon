from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import pandas as pd
import os



class Augmentor:
    def __init__(self):
        self.datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def augment_image(self, image_path, num_augentations=20, label=None):
        """
        Augments a single image.
        Saves {num_augmentations} augmented images to the same directory as the original image with a _ii suffix

        Parameters
        ----------
        image_path : str, required
            Path to the image to augment
        
        num_augmentations : int, optional
            Number of augmentations to perform (default is 20)
        
        label : str, optional
            Label to assign to the augmented images for return statement (default is None)
        
        Returns
        -------
        list
            List of tuples containing the path to the augmented image and the label
        """
        
        img = load_img(image_path)
        img_array = img_to_array(img)
        img_reshaped = img_array.reshape((1,) + img_array.shape)

        img_augmentations = self.datagen.flow(img_reshaped, batch_size=1)

        saved_images = []
        for ii in range(num_augentations):
            img_aug = img_augmentations.next()
            img_aug = img_aug.reshape(img_aug.shape[1:])
            img_aug_path = image_path.replace(".jpg", f"_{ii}.jpg")
            Image.fromarray(img_aug.astype('uint8')).save(img_aug_path)
            saved_images.append((img_aug_path, label))

        return saved_images