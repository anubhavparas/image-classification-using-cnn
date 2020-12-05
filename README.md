# Image Classification using Convolutional Neural Networks
This project aims to classify the images in the given dataset as cats or dogs using convolutional neural networks(CNN)

### Approach and pipeline:
Refer to the [report](Report.pdf) and [code](./Code) for the approach and implementation. 

### Results:
- Results after training 18,000 images of cats and dogs:
    - number of epochs = 15
    - training data / validation data split = 80/20
    - MODEL 
        - CONV 3x3 filter layers with batch norm - **32 x 64 x 96 x 96 x 64**
        - Dense layers with drop out of 0.2 and 0.3 - 256 x 128 x 2 
        - loss: 0.0638 
        - accuracy: 0.9759 
        - val_loss: 0.3255 
        - val_accuracy: 0.9044

- The model was tested on the images in the test1 folder. The performance of the model was very good and was able to predict the animals with 97-99% accuracy.

Plots for model accuracy and loss are following:

![alt text](./output/accuracy_5000images_15epochs.png?raw=true "Model accuracy with 5000 images")

![alt text](./output/loss_5000images_15epochs.png?raw=true "Model loss with 5000 images")

![alt text](./output/accuracy_18000images_15epochs.png?raw=true "Model accuracy with 18000 images")

![alt text](./output/loss_18000images_15epochs.png?raw=true "Model loss with 18000 images")

Classifying the images:

![alt text](./output/cat_prediction1.PNG?raw=true "Cat prediction")

![alt text](./output/cat_prediction2.PNG?raw=true "Cat prediction")

![alt text](./output/dog_prediction1.PNG?raw=true "Dog prediction")

![alt text](./output/dog_prediction2.PNG?raw=true "Dog prediction")



**Output video predicting the images as cats and dogs can be found [here](https://drive.google.com/file/d/1sFOQtDGLc8avs-w5c2NtjmG5i7AYdUk0/view?usp=sharing).**


### Instructions to run the code:
[Input dataset](https://drive.google.com/file/d/19inwa0n1W4DZamjCOm5XAlztvqG_xkjP/view?usp=sharing)

- Go to directory:  _cd Code/_
- To start the training run: 
    - _$ python main.py_

    


