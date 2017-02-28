# Vehicle Detection Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal for this project is to train a Linear SVM classifier to detect vehicles. To do so, I 
performed the following steps:

* Select images labeled as vehicles or "not vehicles" to be used as a training data set 
* Extract features from the training images: 
    1. Histogram of Oriented Gradients (HOG)
    2. Binned color features
    3. Color histograms
* Normalize the extracted features 
* Reserve 20% of the training data to serve as a test set
* Fit a Linear SVM classifier to the data
* Implement a sliding window algorithm to detect vehicles in an image
* On a video stream, combine the sliding window detection with a heatmap and vehicle tracking to estimate 
a bounding box for each detected vehicle

[//]: # (Image References)
[image1]: ./output_images/hsv_hog.png
[image2]: ./output_images/hog_not_car.png
[image3]: ./output_images/hsv_hog_4_pixels_per_cell.png
[image4]: ./output_images/sliding_window_1.png
[image5]: ./output_images/sliding_window_2.png
[image6]: output_images/video_detection_sample.jpg
[project_video]: https://youtu.be/6MdclPVWHzg
[project_video_debug]: https://youtu.be/6MdclPVWHzg

## Vehicle Classifier (<code>VehicleClassifier.py</code>)

This class provides the methods for feature extraction and classifier training.  The user can select 
which features and parameters will be used by the classifier (mind the example in 
<code>train_classifier.py</code>).

I'm using data from the [GTI vehicle image dataset](http://www.gti.ssr.upm.es/data/Vehicle_database.html) 
and the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/) to train the classifier. 
Combined, these datasets provide more than 17000 images for vehicles and non-vehicles.

#### Histogram of Gradients (HOG)

The most important feature for vehicle detection in this implementation is the Histogram of Gradients 
(<code>VehicleClassifier.get_hog_features</code>). A HOG is a feature descriptor that counts occurrences 
of gradient orientations in localized portions of an image.

It is easier to understand how a HOG works by looking at a few examples. The image below shows a 
HOG for a car image in the HSV color space (9 orientations, 8x8 pixels per cell, 2 cells per block):

![HSV HOG][image1]

It is also easy to notice how different the information in a HOG is for a car compared to an image of 
something else:

![Not a car][image2]

We can change how much information we retain in the HOG by tuning the feature extraction parameters. 
For instance, if we set the number of pixels per cell to be 4x4 we get (for the same image and color 
space):

![HSV HOG, 4px][image3]

We can also intuitively notice how some channels seem to provide information that is more 
relevant to the classifier. My goal was to find a set of parameters that were robust enough for 
detection and at the same time did not impose a high performance penalty.

After several experiments, I opted for the following parameters:
* Color space: YCrCb
* HOG orientations: 9
* HOG cells per block: 2 x 2
* HOG pixels per cell: 8 x 8 
* HOG channels Y, Cr and Cb
* Color histogram: all channels, 32 bins
* Spatial binning of color: all channels, 32 x 32

This turned out to be an acceptable compromise between performance and accuracy, though I also 
achieved good results using only two channels for the HOG or by using bigger 
cell sizes (10x10 pixels).

With the selected parameters, the linear SVM classifier achieved a 99.37% test accuracy. The implementation  
details for the classifier training are in <code>VehicleClassifier.train</code>


#### Sliding Window Search

Once we have a robust classifier, we need a strategy to find vehicles in random positions of an image. 
The sliding window search consists of extracting features from successive (sliding) regions of an 
image (a window) and asking the classifier to predict if it is a vehicle or not. More windows usually  
mean a more accurate prediction. Having windows of different sizes also helps identifying vehicles that 
appear in different sizes in the image. 

It is easy to see, however, that having too many windows lead to awful long processing times. After some 
experimentation, I settled for the following strategy:

* Search for 128x128 pixel windows with a 66% overlap over coordinates [0, 720) and [400, 1280)
* Search for 82x82 pixel windows with a 50% overlap over coordinates [0, 720) and [400, 521)

With these parameters, the vehicle detection pipeline can process almost 6 frames per second. 
The code for the sliding window search is in <code>VehicleClassifier.fast_search_vehicles</code>

To help eliminating false positives, positive matches regions are added to a heatmap. 
This proves valuable in the video detection pipeline. Some 
examples of detected boxes and the corresponding heatmap can be seen below:

![Sliding Window 1][image4]
![Sliding Window 2][image5]


## Video Implementation

Here's a link to the [project video][project_video] and to the 
[project video with detailed pipeline info][project_video_debug].

The implementation for vehicle detection can be found at <code>VehicleDetection.py</code>. 

When processing videos, I add the current frame heatmap to a "cooled down" heatmap of the previous 
frames. To discard false positive, I then threshold the heatmap and detect "heat islands" in the frame. 
I can then store vehicle information (centroid and box spans) and estimate the bounding box for 
detected vehicles.

Here's an example of what is used by the vehicle detection on a given frame. The top left image is the 
final result of the frame processing. In the bottom right are the positive detections from the sliding 
window algorithm. The top right is the combined heatmaps of the current frame and cooled down heatmap 
from the previous frames. Finally, the bottom left image displays the heat islands extracted for this 
frame:

![Vehicle Detection][image6]

## Conclusions

There are many ways to improve the current implementation. One straightforward idea to  
increase detection accuracy is to use an ensemble of classifiers, taking advantage of other 
classifier types, color spaces and features.
 
We can easily know which regions of a given image are false positives by looking at the sliding  
window results and the accumulated heatmap. To improve accuracy even further, it's possible to retrain 
the classifier using these false positives (hard negative mining). 
It's also possible to augment the training dataset by using images of the detected vehicles.

Another possible improvement is to use smaller window sizes in the sliding window search. 
This would increase detection for partially visible cars (at the cost of processing time).

A more interesting idea is to use the information we already store about detected vehicles to  
deal with situations where a car is fully or partially occluded by another. It is also possible to 
combine information from previous projects to detect on which lane each vehicle is. Finally, it 
should be possible to estimate the relative speed of each detected vehicle in relation to ours.  

 