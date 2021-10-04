# MNIST
This is a repository for my personal project that includes MNIST Classification accuracy comparison and improvising the model.
# General Deep Learning Model
This model resides in the DL folder. It includes a requirements.txt file which has the requirements for the model. I got a test accuracy of 95.42% for this model. We have used the Pytorch library with a usual DNN with 2 hidden layers and Cross Entropy Loss. The optimizer used here is the normal SGD (Stocastic Gradient Descent) with a batch size of 64. As we can see this model is a bit general and so works with average accuracy.
# Siamese Network
Refer to the notebook siamese.ipynb for this approach. Here, we are using the Siamese Network that has 2 embedding networks to reduce the data size from 784 (28 by 28) to 128. This is fed into another layer that has 2 inputs and it measures the euclidean distance between these inputs. The final (Lambda) layer gives an output which if less than 0.5, the 2 images are similar (belong to same class) otherwise they belong to different classes. I got an accuracy of 97.42% for the testing dataset in this case.
# Further Improvements
I tried to implement using Triplet or Quadruplet Loss but unfortunately that's not working at the moment. Another improvement can be to use Mahalanobis distance metric.
