# Wine Quality Predictor 
# A Nueral Network Project following the Universal Machine Learning Workflow
I attempted to predict a wine's sensory quality (1-10) by its physicochemical properties using NN clasifier, achieving an accuracy of 67.62%

## Code and Resources
* **Python version:** 3.7
* **Preferable development environment:** Colaboratory
* **Packages:** tensorflow, sklearn, numpy, pandas, matplotlib, seaborn
* **Dataset Source:** tensorflow_datasets
* **Universal Machine Learning Workflow:** Deep Learning with Python

## Project Overview
* Developed a NN model that attempts to predict the sensory quality of wine based on its physicochemical properties
* This project follows the Universal Machine Learning Workflow which includes: 
  1. Define the problem
  2. Set the measure of success
  3. Set an evaluation protocol
  4. Load and prepare the data
  5. Develope a model that does better than a baseline model
  6. Develope a model that overfits by scaling up
  7. Tune the hyperparameters, add regularisation and train and test the final model.
* For the scaling up step, I used a DoE strategy to explore the influence of 3 hyperparameters: number of nodes per layer, number of layers, type of activation function. The aim of the DoE is to maximise the exploratory space while minimising the number of experiments.
* The final model achieved an accuracy of 67.62%, improving the initial 44.9% of the baseline model.
* The confusion matrix showed the model confused the original class with a similar class (e.g. 6 with 7) 
* Improvements of the predictor model could be done by changing the model as a regressor or bin classes to form general classes (low, medium, high quality).


## Define the problem
Only wine experts can qualify a wine by its sensory qualities, but there are few of them. On the other hand, wineries can accurately measure the physicochemical properties of wine with analytical equipment. Wineries would be able to develop faster good quality wines if they could predict the quality of wine based on its physicochemical properties.

For this project, I worked with a dataset containing the physicochemical properties of 4898 samples of wines as input and their quality scores (0-9) provided by wine experts. The input data consists of 11 numerical continuous variables while the output data consists of a single discrete numerical value. As the output is discrete and not continuous, the problem can be considered a single label multiclass classification problem.

Two hypotheses were made:
* The physical properties of wine can be used to predict its sensory quality.
* The available dataset is informative enough to learn the relationship between wine properties and its sensory quality.

## Set the measure of success
The accuracy is the measure of success, being a classification problem.

## Set an evaluation protocol
The evaluation protocol is one hold-out validation

## Load and Prepare the data
* The dataset 'wine_quality' was imported from tensorflow datasets.
* The input features were processed into a numpy array and they were standardise by dividing each feature by substracting from mean dividing by its standard deviation.
* The labels were converted into one-hot encoding labels.
* The data was split into training, validation and test sets (70,15,15).

## Develope a model that does better than a baseline model

### The baseline model
The baseline model accuracy was 44.9% based on the dummy classifier from sklearn due to an unbalanced dataset as can be seen in the distribution graph.

![](https://github.com/CarolinaKra/WineQualityPredictor/blob/main/images/distributions.png) 

### The model that beats the baseline model
The simple NN model was built with one dense layer of 30 nodes and relu as activation function and a second output layer with 10 nodes and softmax activation function.
It was trained with 40 epochs and reached a test accuracy of 55.65%.

## Develope a model that overfits by scaling up
I used my Design of Experiments skills to evaluate different models with different hyperparameters, scaling up the initial model. 
There were 3 desired hyperparameters to investigate: number of nodes, number of layers, activation function type
The aim of the DoE is to maximise the exploratory space while minimising the number of experiments.
The experimental space looked as follow:

![alt text](https://github.com/CarolinaKra/WineQualityPredictor/blob/main/images/DoE%20(1).png)

The initial experiment results show the following training and validation loss vs epochs graphs:

![](https://github.com/CarolinaKra/WineQualityPredictor/blob/main/images/DoEresults.png)

A summary of the results is shown in the following table:

![](https://github.com/CarolinaKra/WineQualityPredictor/blob/main/images/doeTable.png)

From these results I decided to continue working with the model with 4 layers and 150 nodes per layer with tanh activation function and the model with 5 layers, 150 nodes per layer with 'relu', because the first has reached the lowest validation loss and the second, the highest validation accuracy

## Tune the hyperparameters, add regularisation and train and test the final model

### Tune the hyperparameters

Based on the previously selected models, I tunned the hyperparameters to create two new experimental models
One with 4 layers, 150 nodes per layer and relu as activation function
The second with 5 layers, 150 nodes per layer and tanh as activation function.

After training these model, I obtained the following results

![](https://github.com/CarolinaKra/WineQualityPredictor/blob/main/images/tableAfterTunning.png)

The two new experiments were successful, as the first reached a lower minimum validation loss and the second reached a higher maximum validation accuracy in comparison to the previous best models. I continued working with both of them to the regularisation step.

### Add regularisation

There are two types of regularisation techniques, the addition of a regulariser for the weight updates and the addition of dropout layers. I carried out experiments where applied a different combination of regularisation techniques to the models we have chosen to work on in the previous step.

These were the training and validation loss vs epochs graph obtained with the different regularisation techniques on the model with relu activation function.

![](https://github.com/CarolinaKra/WineQualityPredictor/blob/main/images/resultsReluAfterReg.png)

The best regularisation technique appeared to be the addition of a dropout layer, however it needed more epochs to be trained on.
The two models were trained again with this regularisation techniques but with more epochs.

All the regularisation experiment results were summarised in this table:

![](https://github.com/CarolinaKra/WineQualityPredictor/blob/main/images/RegTable.png)

The best model which achieved the highest maximum validation accuracy as well as the lowest minimum validation loss was the model with relu, 4 layers, 150 nodes, and dropout regulariser.

After looking at the optimal number of epochs for this model, I joined the training and validation sets into a final training set and trained the final model with the optimal number of epochs.
This model yield a test accuracy of 67.62% and the following confusion matrix

![](https://github.com/CarolinaKra/WineQualityPredictor/blob/main/images/confmatrix.png)

## Conclusions

* The final model has achieved a 67.62% accuracy which is more than 12% from the initial simple model with a single input layer and almost 20% higher from the baseline model. Hence, I created a model with statistical power.
* Looking back at our initial hypotheses, I concluded for the 1st hypothesis that I could predict the wine quality with 67.62% accuracy given the physicochemical properties of the wine. For the second hypothesis, I concluded that the available dataset is partially informative to learn the relationship between the wine properties and its sensory quality.
* On one hand, I have taken the model to the maximum accuracy I could get by exploring different hyperparameters and regularisation techniques. But, on the other hand, the model isn't accurate enough to be proposed as a wine quality predictor to the wineries.
* The confusion matrix showed that the samples which weren't correctly classified, were generally confused with classes that are mostly similar to the original class, from this I can suggest two possible ways to improve the model:
1. Present the problem as a regression model rather than a multiclass classification.
2. Do feature engineering and join a few classes together, for example, labels up to 4 could be considered low quality, classes 5 and 6 as middle quality and 7 and higher to be considered as high quality. However, this should be discussed with a wine expert to understand where to put exactly the threshold levels.
* Additionally, the best way to improve the model would be to get more data.  
