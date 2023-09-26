import PIL
import torch
import torchvision.transforms as T
from diffusers import StableDiffusionInpaintPipeline

# Load the pretrained inpainting model
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)

pipeline = pipeline.to("cuda")

def replace_birds(input_image_path, output_image_path):
    # Open input image and mask
    input_image = Image.open(input_image_path).resize((512, 512))
    mask_image = Image.open("./mask_plot.png").resize((512, 512))
    prompt = "remove the object in the mask area and replace with an eagle"
    output_image = pipeline(prompt=prompt, image=input_image, mask_image=mask_image).images[0]
    output_image.save(output_image_path)
input_image_filenames = [
    "./std_imgs/S_BD/n02843553_1362.JPEG",
    "./std_imgs/S_BD/n02843553_2568.JPEG",
    "./std_imgs/S_BD/n02843553_2686.JPEG",
    "./std_imgs/S_BD/n02843553_4181.JPEG",
    "./std_imgs/S_BD/n02843553_4896.JPEG"
]
for input_filename in input_image_filenames:
    for i in range(10):
        output_filename = input_filename[:-4] + f"-andBirdsReplaced-{i}.jpg"   
        replace_birds(input_filename, output_filename)
