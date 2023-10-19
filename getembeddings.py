from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch

modelname = "openai/clip-vit-base-patch32"

class ClipEmbed:

    def __init__(self) -> None:
        self.model = CLIPModel.from_pretrained(modelname)
        self.processor = AutoProcessor.from_pretrained(modelname)
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)

    def embed_text(self, text):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        text_features = self.model.get_text_features(**inputs)
        # normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # make a list of text features
        # text_features = text_features.tolist()
        return text_features
    
    def embed_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # make a list of image features
        # image_features = image_features.tolist()
        return image_features

if __name__ == '__main__':

    clip = ClipEmbed()

    # Get the text features
    text_features = clip.embed_text(["australian shephard", "pitbull", "white furry dog", "black furry dog", 'cat'])
    print(text_features)

    # Get the image features
    url = "https://pet-uploads.adoptapet.com/4/c/9/748519289.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image_features = clip.embed_image(image)
    print(image_features)

    # Get the similarity score
    similarity = (100. * image_features @ text_features.T).softmax(dim=-1)
    print(similarity)