import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class ClipFuzz():
    def __init__(self, model_path, fuzziness=20, dilation_radius=10, blur_radius=10):
        self.model_path = model_path
        self.processor = CLIPSegProcessor.from_pretrained(self.model_path)
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_path)

        self.fuzziness = fuzziness
        self.blur_radius = blur_radius
        self.dilation_radius = dilation_radius
    
    def fuzzy(self, image, prompt, threshold):
        inputs = self.processor(
            text=prompt, images=image, padding="max_length", return_tensors="pt"
        )

        # predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            preds = outputs.logits

        pred = torch.sigmoid(preds)
        mat = pred.cpu().numpy()
        mask = Image.fromarray(np.uint8(mat * 255), "L")
        mask = mask.convert("RGB")
        mask = mask.resize(image.size, resample=Image.BILINEAR)
        mask = np.array(mask)[:, :, 0]

        # normalize the mask
        mask_min = mask.min()
        mask_max = mask.max()
        mask = (mask - mask_min) / (mask_max - mask_min)

        # threshold the mask
        bmask = mask > threshold
        # zero out values below the threshold
        mask[mask < threshold] = 0

        # Convert the binary mask to uint8
        binary_mask = np.uint8(bmask * 255)

        # Apply dilation to the binary mask
        kernel = np.ones((self.dilation_radius, self.dilation_radius), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        # Apply Gaussian blur to the dilated mask for smooth edges
        smooth_mask = Image.fromarray(dilated_mask)
        smooth_mask = smooth_mask.filter(ImageFilter.GaussianBlur(self.blur_radius))

        # Create a blurred version of the original image
        blurred_image = image.filter(ImageFilter.GaussianBlur(self.fuzziness))

        # Apply the smooth mask to the blurred image
        censored_image = Image.composite(blurred_image, image, smooth_mask)

        return censored_image

if __name__ == "__main__":
    clipfuzz = ClipFuzz(model_path="/home/ubuntu/DATA1/yuzhucai/prestrain_model/CIDAS--clipseg-rd64-refined", dilation_radius=30, blur_radius=0)
    image = Image.open('/home/ubuntu/yuzhucai/ethicallens/results/dd1_1/generated_images/2_0000001_raw_image.jpg')
    image1 = clipfuzz.fuzzy(image, "nude body", 0.2)
    image1.save("/home/ubuntu/yuzhucai/ethicallens/1.jpg")
    # CUDA_VISIBLE_DEVICES=10 python fuzz.py