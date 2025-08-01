# Data Scientist (Academic Project) – Image Classification with Convolutional Neural Networks (CNN)

## Developing a Computer Vision Model in R, TensorFlow, and Keras 3

I entered the field of Computer Vision to tackle a classic but fundamental problem: image classification to distinguish between cats and dogs. My goal was not only to build a working model, but also to demonstrate my ability to handle a complete Deep Learning workflow, facing the challenge of working with a limited dataset to avoid overfitting and build a robust classifier.

My first step was data preparation. I organized the Microsoft dataset into a clean directory structure (training, validation, and testing) and created a manageable subset for efficient training. Next, I built my base model: a classic Convolutional Neural Network (CNN). This network, with a stacked convolutional and pooling layer architecture, helped me establish a performance benchmark.

As expected on a small dataset, this initial model showed clear signs of overfitting, with high training accuracy but stagnant validation accuracy.

My next, and most crucial, step was to directly address overfitting. I implemented a two-pronged strategy:
1. Data Augmentation: I created a preprocessing pipeline that applied random transformations to the training images in real time (rotations, zooms, translations, etc.). This artificially expanded my training dataset, teaching the model to generalize and be invariant to the position or orientation of the subject in the image.
2. Dropout Regularization: I integrated a Dropout layer into my architecture just before the classification layers. By randomly "turning off" a percentage of neurons during each training step, I prevented the model from becoming overly dependent on specific patterns in the training data. The result was a significantly more robust second model. By training it for 100 epochs using the new techniques, I achieved a substantial improvement in validation accuracy and a clear mitigation of overfitting, demonstrating the effectiveness of my approach.

Achievements and Demonstrated Skills:
+ Developed a highly accurate image classifier using Convolutional Neural Networks, capable of effectively distinguishing between two classes.
+ Diagnosed and resolved a critical overfitting problem, demonstrating a deep understanding of the practical challenges of training Deep Learning models with limited data.
+ Implemented advanced Deep Learning techniques such as Data Augmentation to enrich the dataset and Dropout Regularization to improve the model's generalization.
+ Built and managed a complete image data pipeline, from organizing files into directories to converting images to normalized tensors for model consumption.
+ Demonstrated an iterative and methodical approach to model development: created a base model, identified its weaknesses (overfitting), and built a second, improved model by applying problem-specific solutions.
+ Efficiently managed the Keras 3 framework in R to build, compile, train, and save complex deep learning models.
