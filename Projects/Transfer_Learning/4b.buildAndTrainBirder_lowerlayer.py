
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 
import scipy

raw_dataset=tf.data.TFRecordDataset(['Downloads/dataset_assignment2/birds-20-eachOf-358.tfrecords'])
feature_description={'image':tf.io.FixedLenFeature([],tf.string),
                    'birdType':tf.io.FixedLenFeature([],tf.int64)}

def parse_examples(serialized_examples):
    examples=tf.io.parse_example(serialized_examples,feature_description)
    targets=examples.pop('birdType')
    images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(
        examples['image'],channels=3),tf.float32),299,299)
    return images, targets
train_dataset=raw_dataset.map(parse_examples,num_parallel_calls=4)

raw_dataset=tf.data.TFRecordDataset(['Downloads/dataset_assignment2/birds-10-eachOf-358.tfrecords'])
feature_description={'image':tf.io.FixedLenFeature([],tf.string),
                    'birdType':tf.io.FixedLenFeature([],tf.int64)}

def parse_examples(serialized_examples):
    examples=tf.io.parse_example(serialized_examples,feature_description)
    targets=examples.pop('birdType')
    images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(
        examples['image'],channels=3),tf.float32),299,299)
    return images, targets
test_dataset=raw_dataset.map(parse_examples,num_parallel_calls=4)

nToAugment=4
def augmentImages(image,label):
    resized_image=tf.image.resize_with_pad(image,299,299)
    imageL=[resized_image]
    myGen=keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
            width_shift_range=[-0.2,0.2],height_shift_range=0.2,
        brightness_range=[.6,1.0], shear_range=0.0,
        channel_shift_range=0.0, fill_mode='constant', cval=0.0, horizontal_flip=True,
        vertical_flip=True)
    augmented_images=[next(myGen.flow(resized_image)) for _ in range(nToAugment)]
    labels=[label.numpy() for _ in range(nToAugment+1)]
    imageL.extend(augmented_images)
    return imageL, labels
def augmentImagesTF(image,label):
    func=tf.py_function(augmentImages,[image,label],[tf.float32,tf.int64])
    return func
def mySqueeze(x,y):
    return tf.squeeze(x),y
trainPipeAug=train_dataset.batch(1).prefetch(1).map(augmentImagesTF,num_parallel_calls=4)
trainPipeAug=trainPipeAug.unbatch().map(mySqueeze,num_parallel_calls=4).shuffle(512)
trainPipeAug=trainPipeAug.batch(4).prefetch(1)
trainPipeAug.element_spec

def preprocessWithAspectRatio(image,label):
    resized_image=tf.image.resize_with_pad(image,299,299)
    final_image=keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

trainPipeAug=trainPipeAug.map(preprocessWithAspectRatio).batch(4)
validPipe=test_dataset.map(preprocessWithAspectRatio).batch(4)

print("Shape of trainPipeAug: ", trainPipeAug.element_spec)
print("Shape of validPipe: ", validPipe.element_spec)

def fixup_shape(images, labels):
    images.set_shape([None,299,299,3])
    labels.set_shape([None, 1])
    print()
    print("fixup_shape")
    print(tf.shape(images))
    print(tf.shape(labels))
    return images, labels

trainPipeAug = trainPipeAug.map(fixup_shape,)

def preprocessWithAspectRatio(image,label):
    resized_image=tf.image.resize_with_pad(image,299,299)
    final_image=keras.applications.xception.preprocess_input(resized_image)
    return final_image, label
trainPipe=train_dataset.map(preprocessWithAspectRatio,num_parallel_calls=2).batch(8)
testPipe=test_dataset.map(preprocessWithAspectRatio,num_parallel_calls=2).batch(8)

traindataset=train_dataset.map(preprocessWithAspectRatio,num_parallel_calls=8).batch(8).cache()
testdataset= test_dataset.map(preprocessWithAspectRatio,num_parallel_calls=8).batch(8).cache()

model = keras.models.load_model('birder.h5')

for layer in model.layers:
    layer.trainable=True

model.summary()

earlyStop_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
ss=5e-1
ss=3e-2
optimizer=keras.optimizers.SGD(learning_rate=ss)
checkpoint_cb=keras.callbacks.ModelCheckpoint('birder.h5',
                save_best_only=True)
model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,
                metrics=["accuracy"])
model.fit(trainPipe,validation_data=testPipe,epochs=25,
            callbacks=[checkpoint_cb,earlyStop_cb])


model.evaluate(testPipe)


top2err=tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)
model.compile(loss="sparse_categorical_crossentropy",
optimizer=optimizer,metrics=[top2err])
model.evaluate(testPipe)
