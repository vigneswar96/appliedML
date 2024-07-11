###########################################README#########################################################

# 1) subselectImages: This folder contains 2 python files one is for bird and birdfeeder. Whether the given input image is having both bird and bird feeder then it returns the image. 

	1) subSelectImages_Bird_birdfeeder.py 
	2) subSelectImages_squirrel_birdfeeder.py
    
Image 0 <br>
<img src="https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/bird-1.jpeg" width="250" alt="Bird Feeder"> <br>

Image 0 Mask 

<img src = "https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/masked_bird-1.jpeg" width ="250" alt ="Bird Feeder">

Image 0 Replaced

<img src = "https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/masked_bird-1-birdsReplaced.jpeg" width = "250" alt = "Bird Feeder">

Image 1

<img src="https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/bird-2.jpeg" width="250" alt="Bird Feeder">

Image 1 Mask

<img src="https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/masked_bird-2.jpeg" width="250" alt="Bird Feeder">

Image 1 Replaced

<img src = "https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/masked_bird-2-birdsReplaced.jpeg" width = "250" alt = "Bird Feeder">

This looks a bit wacky :D


Image 2

<img src ="https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/bird-3.jpeg" width="250" alt = "Bird Feeder">

Image 2 Mask 

<img src = "https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/masked_bird-3.jpeg" width="250" alt = "Bird Feeder">

Image 2 Replaced

<img src = "https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/masked_bird-3-birdsReplaced.jpeg" width = "250" alt = "Bird Feeder">

Image 3

<img src ="https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/bird-4.jpeg" width ="250" alt = "Bird Feeder"> 

Image 3 Mask

<img src = "https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/masked_bird-4.jpeg" width = "250" alt = "Bird Feeder">

Image 3 Replaced

<img src = "https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/masked_bird-4-birdsReplaced.jpeg" width = "250" alt = "Bird Feeder">

Image 4

<img src ="https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/bird-5.jpeg" width ="250" alt = "Bird Feeder"> 

Image 4 Mask 

<img src ="https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/masked_bird-5.jpeg" width ="250" alt = "Bird Feeder">

Image 4 Replaced

<img src = "https://github.com/vigneswar96/appliedML/blob/main/Generative_AI/Notebooks/images/masked_bird-5-birdsReplaced.jpeg" width= "250" alt= "Bird Feeder">



# 2) Segmentation: This folder contains all files related to Segmentation of Images.
   
	What Am i Segmenting?
	Well, Importing Libraries and Initializing Models:

	Several libraries are imported, including Transformers and PyTorch. Multiple models are initialized, such as SamModel and YOLO (You Only Look Once) for object detection.
	Display Functions:

	Functions like show_mask, show_points, and show_box are defined to display masks, points, and bounding boxes on images, respectively.
	Input Data Preparation:

	File paths for input images are taken as user input and split into a list. Images are loaded and prepared for further processing.
	Object Detection using YOLO:

	YOLO model is used to perform object detection on the input images. Detected objects are displayed on the images.
	Instance Segmentation using SAM (Segment Anything Model):

	The SAM model is used for instance segmentation. The model predicts masks for the detected objects.
	Visualization of Masks:

	The predicted masks for the detected objects are visualized and saved as images. 
	Segmentation works for both squirrels and birds. 

4) Removesquirrels: 

	Importing Libraries and Loading Pre-trained Models:

	The code starts by importing necessary libraries such as PIL, requests, torch, etc.
	It then loads a pre-trained Stable Diffusion Inpainting model using the StableDiffusionInpaintPipeline from RunwayML.
	Preparing Images:

	Images (a mask image and a raw image) are loaded using the PIL library.
	The mask image is loaded from "./mask_plot.png" and the raw image from "./std_imgs/S_BD/n02843553_1362.JPEG".
	Image Transformations:

	The mask image is transformed to tensor using torchvision's ToTensor() and then converted back to a PIL image.
	The initial image and mask image are resized to (512, 512) using PIL.
	Inpainting with Stable Diffusion Model:

	A prompt is defined to instruct the model on how to inpaint the image.
	The pipeline is used to inpaint the image based on the given prompt, initial image, and mask image.
	The inpainted image is displayed using matplotlib.
	Replacing Objects in Images:

	A function replace_birds is defined to replace objects in an input image with birds using the inpainting model.
	A list of input image filenames is provided, and for each input image, the function is called to replace objects with birds. It saves multiple output images with variations.

# 5) ReplaceBirds: 
	The images are to be replaced with birds. 
	
# 6) GenerateBirdFeederImagesfromText: 
	Importing Libraries and Loading the Pretrained Model:

	Import the necessary libraries: from diffusers import StableDiffusionPipeline for using the Stable Diffusion model and torch for deep learning operations.
	Specify the model identifier (model_id) for the pre-trained Stable Diffusion model to be used (runwayml/stable-diffusion-v1-5).
	Load the pre-trained Stable Diffusion model using from_pretrained, specifying the desired torch data type (torch.float16).
	Move the model to the CUDA device for GPU acceleration.
	Defining the Prompt:

	Define a prompt describing the desired image: "a photo of a bird looking at a birdfeeder". This prompt will guide the model in generating an image based on the given description.
	Generating the Image:

	Use the pipe object (the Stable Diffusion model) to generate an image based on the provided prompt.
	Retrieve the generated image from the model output.
	Saving the Image:

	Save the generated image to a file named "bird_and_birdfeeder.png" using the save method provided by the Image object.

  <img src="salome-guruli-ST_QmOkfdIA-unsplash.jpg" width="350" title="Bird Feeder" />

