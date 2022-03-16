# Wine Quality Predictor 
# A Nueral Network Project following the Universal Machine Learning Workflow
I attempted to predict a wine's sensory quality (1-10) by its physicochemical properties using NN clasifier, achieving an accuracy of 67.62%

## Code and Resources
* **Python version:** 3.7
* **Preferable development environment:** Colaboratory
* **Code and Packages:** tensorflow, sklearn, numpy, pandas, matplotlib, seaborn
* **Dataset Source:** tensorflow_datasets

## Project Overview
* Developed a NN model that attempts to predict the sensory quality of wine based on its physicochemical properties
* This project follows the Universal Machine Learning Workflow which includes: 
  1. Define the problem
  2. Set the measure
  3. Set an evaluation protocol
  4. Load and prepare the data
  5. Develope a model that does better than a baseline model
  6. Develope a model that overfits by scaling up
  7. Tune the hyperparameters, add regularisation and train and test the final model.
* For the scaling up step, I used a DoE strategy to explore the influence of 3 hyperparameters: number of nodes per layer, number of layers, type of activation function. The aim of the DoE is to maximise the exploratory space while minimising the number of experiments.
* The final model achieved an accuracy of 67.62%, improving the initial 44.9% of the baseline model.
* The confusion matrix showed the model confused the original class with a similar class (e.g. 6 with 7) 
* Improvements of the predictor model could be done by changing the model as a regressor or bin classes to form general classes (low, medium, high quality).


   
