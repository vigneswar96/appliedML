import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 

train_dataset=tf.data.TFRecordDataset(['Downloads/dataset_assignment2/birds-vs-squirrels-train.tfrecords']) # tfrecord file path    
feature_description={'image':tf.io.FixedLenFeature([],tf.string),   # tfrecord file feature description
                     'label':tf.io.FixedLenFeature([],tf.int64)}    # tfrecord file feature description
def parse_examples(serialized_examples):
    examples=tf.io.parse_example(serialized_examples,feature_description) 
    targets=examples.pop('label')
    images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(
        examples['image'],channels=3),tf.float32),299,299)
    return images, targets


cnt=1
plt.figure(figsize=(32,16))
for inst in train_dataset.take(4*8):
    plt.subplot(4,8,cnt)
    plt.imshow(inst[0].numpy()/255)
    label_value = inst[1].numpy()
    plt.title('Label: {}'.format(label_value))
    cnt=cnt+1 


test_dataset=tf.data.TFRecordDataset(['Downloads/dataset_assignment2/birds-vs-squirrels-validation.tfrecords'])# tfrecord file path
feature_description={'image':tf.io.FixedLenFeature([],tf.string),
                     'label':tf.io.FixedLenFeature([],tf.int64)}
def parse_examples(serialized_examples):
    examples=tf.io.parse_example(serialized_examples,feature_description)
    targets=examples.pop('label')
    images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(
        examples['image'],channels=3),tf.float32),299,299)
    return images, targets
test_dataset=test_dataset.map(parse_examples,num_parallel_calls=2)

cnt=1
plt.figure(figsize=(32,16))
for inst in test_dataset.take(4*8):
    plt.subplot(4,8,cnt)
    plt.imshow(inst[0].numpy()/255)
    label_value = inst[1].numpy()
    plt.title('Label: {}'.format(label_value))
    cnt=cnt+1  

