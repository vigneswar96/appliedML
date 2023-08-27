#############################################################Brief Overview###############################################################################

The project is about implementing a transfer learning technique. 
Here we use Xception architecture from Keras applications to build a deep learning model for Image classification. 

Objective -------->  To Classify the Image is whether an image of Squirrel or Bird. 

Method    -------->  Here we have used Xception which is trained on Imagenet database. There are two subprograms available and explained as below. 

					 4a.buildAndTrainBirder_upperlayer.py --> Keeping the lower layers frozen and training only the top layer which gives a decent performance. 
					 4b.buildAndTrainBirder_lowerlayer.py --> Keeping the top layer weights constant retraining the lower layers to adjust their weight accordingly.

Results   -------->  Obtained a  validation accuracy of 97.55% and a validation loss of 0.1096. 

Challenge ------->  The Major challenege is to get the tf recordsfile.( I have uploaded the tfrecords files in tfrecords folder. 

What is tfrecords ? 
Tensorflow records is a binary format used in TensorFlow to store and efficiently manage large datasets. TFRECORDS store data as sequence of binary records, each containing serialized data. 
TFRecords are widely used in the TensorFlow ecosystem because they provide a performance and flexible way to manage and process data for machine learning tasks. They are especially popular when dealing with large-scale image datasets.

###########################################################################################################################################################
