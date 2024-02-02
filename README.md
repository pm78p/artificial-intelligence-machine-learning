---
__Files in Branch:__

- __[Machine learning](https://github.com/pm78p/root/tree/main/Machine%20learning)__ - Here, there are number of projects which are related to ML field.
- __[computer vision](https://github.com/pm78p/root/tree/main/computer%20vision)__ - Projects are related to Computer Vision field.
- __[deep learning](https://github.com/pm78p/root/tree/main/deep%20learning)__ - Projects are related to DL field.
- __[kaggle competion](https://github.com/pm78p/root/tree/main/kaggle%20competion)__ - Project is related to Kaggle competition.
  

### [Machine learning](https://github.com/pm78p/root/tree/main/Machine%20learning)


[1-ICU italy dataset](https://github.com/pm78p/root/tree/main/Machine%20learning/1-ICU%20italy%20dataset) - Here we modeled ICU italy dataset with help of the some methods
(Multy layer perceptron, Random forest, ada boost, linear regression). The results for each methods listed below:

![results](https://github.com/pm78p/root/blob/main/1-icu_result.png)

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/1-ICU%20italy%20dataset/codes.ipynb)

[2-sarcasm detection dataset](https://github.com/pm78p/root/tree/main/Machine%20learning/2-sarcasm%20detection%20dataset) - Here After doing some Data visualization and  Data
preprocessing we modeled sarcasm on reddit dataset from kaggle with help of the some methods.
In one way we use TF-IDF embedding method, then we train, a. Logistic Regression - b. SVM - c. Ada Boost, models on data. 
In other way we embedd text with the help of the Word2Vec and train RNN model.
The results for each methods listed below:

![results](https://github.com/pm78p/root/blob/main/2-sarcasm_result.png)

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/2-sarcasm%20detection%20dataset/codes.ipynb)

[3-data cleaning & EDA & Dimension Reduction](https://github.com/pm78p/root/tree/main/Machine%20learning/3-data%20cleaning%20%26%20EDA%20%26%20Dimension%20Reduction) -

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/3-data%20cleaning%20%26%20EDA%20%26%20Dimension%20Reduction/codes.ipynb)

[4-regression](https://github.com/pm78p/root/tree/main/Machine%20learning/4-regression) - Here we use "The Boston Housing" Dataset. Fisrt, we compare gradient descent and stochastic gradient descent on this dataset. Next, a. implement 1st order regression with SSE
as cost function - b. 3rd order regression with SSE as cost function - c. 3rd order regression with SSE as cost function and a regularization term and predict test data.
Last, we use a 10-fold cross validation.

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/4-regression/codes.ipynb)

[5-logistic reg & ada boost](https://github.com/pm78p/root/tree/main/Machine%20learning/5-logistic%20reg%20%26%20ada%20boost) - Here we use "Blood Transfusion Service Center"
dataset. First, we train logistic regression classiﬁer and a decision tree to classify the dataset. Then, fine tuning models. Finally, Train an AdaBoost classiﬁer that each weak 
learner is a stump (Stumps are decision trees with depth one).

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/5-logistic%20reg%20%26%20ada%20boost/codes.ipynb)

[6-probabilistic graphical models](https://github.com/pm78p/root/tree/main/Machine%20learning/6-probabilistic%20graphical%20models) - Here we try to use probabilistic graphical models
to denoising messages in a related dataset

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/6-probabilistic%20graphical%20models/codes.ipynb)

[7-clustering k-means & EM & GMM](https://github.com/pm78p/root/tree/main/Machine%20learning/7-clustering%20k-means%20%26%20EM%20%26%20GMM) - Here we made a random dataset via sklearn 
by make_classification function in this library. Then, implementing k-means and GMM to classify the datas.

![results](https://github.com/pm78p/root/blob/main/7-clusterin_result.png)

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/7-clustering%20k-means%20%26%20EM%20%26%20GMM/codes.ipynb)

[8-CNN](https://github.com/pm78p/root/tree/main/Machine%20learning/8-CNN) - 

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/8-CNN/codes.ipynb)

### [computer vision](https://github.com/pm78p/root/tree/main/computer%20vision)
[1-panaromic image](https://github.com/pm78p/artificial-intelligence-machine-learning) - Here, provided Python script executes an image stitching process to create a panorama from multiple images. It uses the SIFT feature detection algorithm to find keypoints and descriptors in overlapping image regions. The RANSAC algorithm is utilized to robustly estimate homography matrices despite outliers, aligning images relative to a central reference. The script then warps and blends the images into a single panoramic output. It includes a function for autocropping to remove the black edges resulting from the warping process, optimizing the visual result. This script effectively demonstrates computer vision techniques to stitch images for applications like panoramic photography.

[Input](https://github.com/pm78p/artificial-intelligence-machine-learning/tree/main/computer%20vision/first%20step/panaroma/input%20data):
8 different pictures in different unknown angels.

<p align="center">
  <img src="https://github.com/pm78p/artificial-intelligence-machine-learning/blob/main/computer%20vision/first%20step/panaroma/input%20data/03.JPG" width="100" alt="Image 1">
  <img src="https://github.com/pm78p/artificial-intelligence-machine-learning/blob/main/computer%20vision/first%20step/panaroma/input%20data/04.JPG" width="100" alt="Image 2">
  <img src="https://github.com/pm78p/artificial-intelligence-machine-learning/blob/main/computer%20vision/first%20step/panaroma/input%20data/05.JPG" width="100" alt="Image 3">
  <img src="https://github.com/pm78p/artificial-intelligence-machine-learning/blob/main/computer%20vision/first%20step/panaroma/input%20data/06.JPG" width="100" alt="Image 4">
  <img src="https://github.com/pm78p/artificial-intelligence-machine-learning/blob/main/computer%20vision/first%20step/panaroma/input%20data/07.JPG" width="100" alt="Image 5">
  <img src="https://github.com/pm78p/artificial-intelligence-machine-learning/blob/main/computer%20vision/first%20step/panaroma/input%20data/08.JPG" width="100" alt="Image 6">
  <img src="https://github.com/pm78p/artificial-intelligence-machine-learning/blob/main/computer%20vision/first%20step/panaroma/input%20data/09.JPG" width="100" alt="Image 7">
  <img src="https://github.com/pm78p/artificial-intelligence-machine-learning/blob/main/computer%20vision/first%20step/panaroma/input%20data/10.JPG" width="100" alt="Image 8">
  <!-- Add more images as needed -->
</p>


The results:

![results](https://github.com/pm78p/artificial-intelligence-machine-learning/blob/main/computer%20vision/first%20step/panaroma/r13_crop.jpg)

*** [Code](https://github.com/pm78p/artificial-intelligence-machine-learning/blob/main/computer%20vision/first%20step/panaroma/codes.py)

2- ......

!!!sorry, incomplete list
### [deep learning](https://github.com/pm78p/root/tree/main/deep%20learning)
!!!sorry, incomplete list
### [kaggle competion](https://github.com/pm78p/root/tree/main/kaggle%20competion)
!!!sorry, incomplete list
