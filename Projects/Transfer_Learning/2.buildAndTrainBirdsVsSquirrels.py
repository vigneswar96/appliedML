
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 


import tensorflow as tf
train_dataset=tf.data.TFRecordDataset(['Downloads/dataset_assignment2/birds-vs-squirrels-train.tfrecords'])
feature_description={'image':tf.io.FixedLenFeature([],tf.string),
                     'label':tf.io.FixedLenFeature([],tf.int64)}
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


test_dataset=tf.data.TFRecordDataset(['Downloads/dataset_assignment2/birds-vs-squirrels-validation.tfrecords'])
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


base_model=keras.applications.xception.Xception(weights='imagenet',include_top=True)


base_model.summary()


base_model=keras.applications.xception.Xception(weights='imagenet',include_top=False)
base_model.summary()



base_model=keras.applications.xception.Xception(weights='imagenet',include_top=False)



avg=keras.layers.GlobalAveragePooling2D()(base_model.output)
output=keras.layers.Dense(3,activation="softmax")(avg)
model=keras.models.Model(inputs=base_model.input,outputs=output)
model.summary()



def preprocessWithAspectRatio(image,label):
    resized_image=tf.image.resize_with_pad(image,299,299)
    final_image=keras.applications.xception.preprocess_input(resized_image)
    return final_image, label
trainPipe=train_dataset.map(preprocessWithAspectRatio,num_parallel_calls=2).batch(16)
testPipe=test_dataset.map(preprocessWithAspectRatio,num_parallel_calls=2).batch(16)



for layer in base_model.layers:
    layer.trainable=False
for layer in model.layers:
    print(layer.trainable)



checkpoint_cb=keras.callbacks.ModelCheckpoint('model.h5',
save_best_only=True)


earlyStop_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
ss=5e-1
optimizer=keras.optimizers.SGD(learning_rate=ss)
model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,
metrics=["accuracy"])


model.fit(trainPipe,validation_data=testPipe,epochs=25,callbacks=[checkpoint_cb,earlyStop_cb])








