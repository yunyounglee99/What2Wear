from PIL import Image, ImageOps, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel

class ProcessImage:
  def __init__(self, model, processor):
    self.model = model
    self.processor = processor
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model.to(self.device)
    self.categories = ["Top", "Bottom", "Shoes"]
    self.input_dim = 512
    self.output_dim = 128

  def add_equal_white_padding(image, threshold = 300):
    gray_img = image.convert("L")
    np_img = np.array(gray_img)

    non_white = np.where(np_img < threshold)

    if non_white[0].size == 0 or non_white[1].size == 0:
      return image
    
    top, bottom = non_white[0][0], non_white[0][-1]
    left, right = non_white[1][0], non_white[1][-1]

    cropped_img = image.crop((left, top, right, bottom))

    padding_vertical = max(0, (image.height - (bottom - top)) // 2)
    padding_horizontal = max(0, (image.width - (right - left)) // 2)

    final_img = ImageOps.expand(
      cropped_img,
      border = (padding_horizontal, padding_vertical),
      fill = "white"
    )

    return final_img
  
  def crop_and_embed_image(self, top_height, bottom_height, image):
    segment_height = image.height // 20

    top_crop = top_height * segment_height
    bottom_crop = bottom_height * segment_height
    cropped_img = image.crop((0, top_crop, image.width, bottom_crop))

    with torch.no_grad():
      inputs = self.processor(images = cropped_img, returrn_tensors = "pt", padding = True)
      inputs = {k: v.to(self.device) for k, v in inputs.items()}
      embedding = self.model.get_image_features(**inputs)
      embedding = embedding / embedding.norm(dim = -1, keepdim = True)

    return cropped_img, embedding
  
  def process_image(self, image_path):
    results = {}
    original_image = Image.open(image_path)

    for category in self.categories:
      if category == "Top":
        top_height = 4
        bottom_height = 12
      elif category == "Bottom":
        top_height = 8
        bottom_height = 17
      elif category == "Shoes":
        top_height = 14
        bottom_height = 18
      cropped_image, embedding = self.crop_and_embed_image(top_height, bottom_height, original_image)
      results[category] = {
        'cropped_image' : cropped_image,
        'embedding' : embedding
      }

      return results
    
  def generate_outfit_vector(self, results):
    top_result = results.get("Top")
    bottom_result = results.get("Bottom")
    shoes_result = results.get("Shoes")

    top_embedding = top_result["embedding"].squeeze() if top_result and "embedding" in top_result else torch.zeros(self.input_dim, device = self.device)
    bottom_embedding = bottom_result["embedding"].squeeze() if bottom_result and "embedding" in bottom_result else torch.zeros(self.input_dim, device=self.device)
    shoes_embedding = shoes_result["embedding"].squeeze() if shoes_result and "embedding" in shoes_result else torch.zeros(self.input_dim, device=self.device)

    outfit_vector = torch.stack([top_embedding, bottom_embedding, shoes_embedding])

    return outfit_vector
  
"""
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
process_image = ProcessImage(model, processor)
"""
