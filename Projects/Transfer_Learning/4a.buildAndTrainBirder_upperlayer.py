
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 



raw_dataset=tf.data.TFRecordDataset(['Downloads/dataset_assignment2/birds-20-eachOf-358.tfrecords'])
feature_description={'image':tf.io.FixedLenFeature([],tf.string),
                    'birdType':tf.io.FixedLenFeature([],tf.int64)}

def parse_examples(serialized_examples):
    examples=tf.io.parse_example(serialized_examples,feature_description)
    targets=examples.pop('birdType')
    images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(
        examples['image'],channels=3),tf.float32),299,299)
    return images, targets
train_dataset=raw_dataset.map(parse_examples,num_parallel_calls=2)



batched_train_dataset = train_dataset.batch(32)
for batch in batched_train_dataset:
    print(batch[0].shape)
    break

raw_dataset=tf.data.TFRecordDataset(['Downloads/dataset_assignment2/birds-10-eachOf-358.tfrecords'])
feature_description={'image':tf.io.FixedLenFeature([],tf.string),
                    'birdType':tf.io.FixedLenFeature([],tf.int64)}

def parse_examples(serialized_examples):
    examples=tf.io.parse_example(serialized_examples,feature_description)
    targets=examples.pop('birdType')
    images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(
        examples['image'],channels=3),tf.float32),299,299)
    return images, targets
test_dataset=raw_dataset.map(parse_examples,num_parallel_calls=2)


base_model=keras.applications.xception.Xception(weights='imagenet',include_top=True)


base_model.summary()

base_model=keras.applications.xception.Xception(weights='imagenet',include_top=False)
base_model.summary()


avg=keras.layers.GlobalAveragePooling2D()(base_model.output)
output=keras.layers.Dense(358,activation="softmax")(avg)
model=keras.models.Model(inputs=base_model.input,outputs=output)
model.summary()


def preprocessWithAspectRatio(image,label):
    resized_image=tf.image.resize_with_pad(image,299,299)
    final_image=keras.applications.xception.preprocess_input(resized_image)
    return final_image, label
trainPipe=train_dataset.map(preprocessWithAspectRatio,num_parallel_calls=2).batch(16)
testPipe=test_dataset.map(preprocessWithAspectRatio,num_parallel_calls=2).batch(16)


traindataset=train_dataset.map(preprocessWithAspectRatio,num_parallel_calls=16).batch(16).cache()
testdataset= test_dataset.map(preprocessWithAspectRatio,num_parallel_calls=16).batch(16).cache()


for layer in base_model.layers:
    layer.trainable=False
for layer in model.layers:
    print(layer.trainable)



checkpoint_cb=keras.callbacks.ModelCheckpoint('birder.h5',
save_best_only=True)



earlyStop_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
ss=5e-1
optimizer=keras.optimizers.SGD(learning_rate=ss)
model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,
metrics=["accuracy"])



model.fit(trainPipe,validation_data=testPipe,epochs=25,callbacks=[checkpoint_cb,earlyStop_cb])


model=tf.keras.models.load_model('birder.h5')



top5err=tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5,name='top5')
top10err=tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10,name='top10')
top20err=tf.keras.metrics.SparseTopKCategoricalAccuracy(k=20,name='top20')
optimizer=keras.optimizers.SGD(learning_rate=ss)



model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,
              metrics=['accuracy',top5err,top10err,top20err])



resp=model.evaluate(testPipe)





