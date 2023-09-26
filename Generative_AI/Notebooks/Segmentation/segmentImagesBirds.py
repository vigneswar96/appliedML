from transformers import AutoImageProcessor, AutoModelForObjectDetection
import torch
from transformers import SamModel, SamProcessor
from transformers import pipeline
import matplotlib.pyplot as plt 
from segment_anything import sam_model_registry, SamPredictor 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
yolo_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
yolo_model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    from PIL import Image

file_paths_input = input("Enter a comma-separated list of file paths: ")
file_paths = file_paths_input.split(",")

raw_images = []
for file_path in file_paths:
    raw_image = Image.open(file_path.strip(), 'r')
    raw_images.append(raw_image)

inputs = yolo_processor(images=raw_images[0], return_tensors="pt")
outputs = yolo_model(**inputs)

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

target_sizes = torch.tensor([raw_images[0].size[::-1]])
results = yolo_processor.post_process_object_detection(outputs, threshold=0.2, target_sizes=target_sizes)[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {yolo_model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
    show_boxes_on_image(raw_images[0], [box])
    plt.savefig('./raw_image_0_box.png')
inputs1 = yolo_processor(images=raw_images[1], return_tensors="pt")
outputs1 = yolo_model(**inputs1)

target_sizes = torch.tensor([raw_images[1].size[::-1]])
results1 = yolo_processor.post_process_object_detection(outputs1, threshold=0.2, target_sizes=target_sizes)[0]
for score, label, box in zip(results1["scores"], results1["labels"], results1["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {yolo_model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
    show_boxes_on_image(raw_images[1], [box])

inputs2 = yolo_processor(images=raw_images[2], return_tensors="pt")
outputs2 = yolo_model(**inputs2)

target_sizes = torch.tensor([raw_images[2].size[::-1]])
results2 = yolo_processor.post_process_object_detection(outputs2, threshold=0.2, target_sizes=target_sizes)[0]
for score, label, box in zip(results2["scores"], results2["labels"], results2["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {yolo_model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
    show_boxes_on_image(raw_images[2], [box])

inputs3 = yolo_processor(images=raw_images[3], return_tensors="pt")
outputs3 = yolo_model(**inputs3)

target_sizes = torch.tensor([raw_images[3].size[::-1]])
results3 = yolo_processor.post_process_object_detection(outputs3, threshold=0.2, target_sizes=target_sizes)[0]
for score, label, box in zip(results3["scores"], results3["labels"], results3["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {yolo_model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
    show_boxes_on_image(raw_images[3], [box])

inputs4  = yolo_processor(images=raw_images[4], return_tensors="pt")
outputs4 = yolo_model(**inputs4)

target_sizes = torch.tensor([raw_images[4].size[::-1]])
results4 = yolo_processor.post_process_object_detection(outputs4, threshold=0.2, target_sizes=target_sizes)[0]
for score, label, box in zip(results4["scores"], results4["labels"], results4["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {yolo_model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
    show_boxes_on_image(raw_images[4], [box])

inputs5 = yolo_processor(images=raw_images[5], return_tensors="pt")
outputs5 = yolo_model(**inputs5)

target_sizes = torch.tensor([raw_images[5].size[::-1]])    
results5 = yolo_processor.post_process_object_detection(outputs5, threshold=0.2, target_sizes=target_sizes)[0]
for score, label, box in zip(results5["scores"], results5["labels"], results5["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {yolo_model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
    show_boxes_on_image(raw_images[5], [box])

input_boxes = [np.ndarray.tolist(results["boxes"].detach().numpy())]
inputs = sam_processor(raw_image[0], input_boxes=input_boxes, return_tensors="pt").to(device)
image_embeddings = sam_model.get_image_embeddings(inputs["pixel_values"])

inputs.pop("pixel_values", None)
inputs.update({"image_embeddings": image_embeddings})

with torch.no_grad():
    outputs = sam_model(**inputs)

masks = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
scores = outputs.iou_scores

input_boxes1 = [np.ndarray.tolist(results1["boxes"].detach().numpy())]
inputs1 = sam_processor(raw_images[1], input_boxes=input_boxes1, return_tensors="pt").to(device)
image_embeddings = sam_model.get_image_embeddings(inputs1["pixel_values"])

inputs1.pop("pixel_values", None)
inputs1.update({"image_embeddings": image_embeddings})

with torch.no_grad():
    outputs = sam_model(**inputs1)

masks1 = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs1["original_sizes"].cpu(), inputs1["reshaped_input_sizes"].cpu())
scores1 = outputs.iou_scores

input_boxes2 = [np.ndarray.tolist(results2["boxes"].detach().numpy())]
inputs2 = sam_processor(raw_images[2], input_boxes=input_boxes2, return_tensors="pt").to(device)
image_embeddings = sam_model.get_image_embeddings(inputs2["pixel_values"])

inputs2.pop("pixel_values", None)
inputs2.update({"image_embeddings": image_embeddings})

with torch.no_grad():
    outputs = sam_model(**inputs2)

masks2 = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs2["original_sizes"].cpu(), inputs2["reshaped_input_sizes"].cpu())
scores2 = outputs.iou_scores

input_boxes3 = [np.ndarray.tolist(results3["boxes"].detach().numpy())]
inputs3 = sam_processor(raw_images[3], input_boxes=input_boxes3, return_tensors="pt").to(device)
image_embeddings = sam_model.get_image_embeddings(inputs3["pixel_values"])

inputs3.pop("pixel_values", None)
inputs3.update({"image_embeddings": image_embeddings})

with torch.no_grad():
    outputs = sam_model(**inputs3)

masks3 = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs3["original_sizes"].cpu(), inputs3["reshaped_input_sizes"].cpu())
scores3 = outputs.iou_scores

input_boxes4 = [np.ndarray.tolist(results4["boxes"].detach().numpy())]
inputs4 = sam_processor(raw_images[4], input_boxes=input_boxes4, return_tensors="pt").to(device)
image_embeddings = sam_model.get_image_embeddings(inputs4["pixel_values"])

inputs4.pop("pixel_values", None)
inputs4.update({"image_embeddings": image_embeddings})

with torch.no_grad():
    outputs = sam_model(**inputs4)

masks4 = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs4["original_sizes"].cpu(), inputs4["reshaped_input_sizes"].cpu())
scores4 = outputs.iou_scores

input_boxes5 = [np.ndarray.tolist(results5["boxes"].detach().numpy())]
inputs5 = sam_processor(raw_images[5], input_boxes=input_boxes5, return_tensors="pt").to(device)
image_embeddings = sam_model.get_image_embeddings(inputs5["pixel_values"])

inputs5.pop("pixel_values", None)
inputs5.update({"image_embeddings": image_embeddings})

with torch.no_grad():
    outputs = sam_model(**inputs5)

masks5 = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs5["original_sizes"].cpu(), inputs5["reshaped_input_sizes"].cpu())
scores5 = outputs.iou_scores

mask_plot = []
for index, mask in enumerate(masks[0]):
    mask_plot = (mask_plot == True) | (mask[:, :, :] == True)

mask_plot = mask_plot.float()
plt.imshow(mask_plot.permute(1, 2, 0))
plt.savefig('./mask_plot.png')

mask_plot1 = []
for index, mask in enumerate(masks1[0]):
    mask_plot1 = (mask_plot1 == True) | (mask[:, :, :] == True)

mask_plot1 = mask_plot1.float()
plt.imshow(mask_plot1.permute(1, 2, 0))
plt.savefig('./mask_plot1.png')

mask_plot3 = []
for index, mask in enumerate(masks3[0]):
    mask_plot3 = (mask_plot3 == True) | (mask[:, :, :] == True)

mask_plot3 = mask_plot3.float()
plt.imshow(mask_plot3.permute(1, 2, 0))
plt.savefig('./mask_plot3.png')

mask_plot4 = []
for index, mask in enumerate(masks4[0]):
    mask_plot4 = (mask_plot4 == True) | (mask[:, :, :] == True)

mask_plot4 = mask_plot4.float()
plt.imshow(mask_plot4.permute(1, 2, 0))
plt.savefig('./mask_plot4.png')

mask_plot5 = []
for index, mask in enumerate(masks5[0]):
    mask_plot5 = (mask_plot5 == True) | (mask[:, :, :] == True)

mask_plot5 = mask_plot5.float()
plt.imshow(mask_plot5.permute(1, 2, 0))
plt.savefig('./mask_plot5.png')




