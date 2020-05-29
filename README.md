# machine-learning-classes
A short project where I created 2 classes which combine many of the scikit-learn tools for both regression and classification tasks in order to simplify the process of creating and saving model results. In addition, I have integrated many visualisation and measurement functions within the class.

## Classification Class
This class is for classification problems. The below scikit-learn models are included:
- K Nearest Neighbors
- Logistic regression
- Decision Trees
- Random Forest
- ADA boosting
- Gradient Boosting
- Naive Bayes
- Linear Support Vectors
- Polynomial Support Vectors
- Gaussian Support Vectors
- Neural Networks - this is the only one which doesn't run automatically when you select run_all=True. You need to run this seperately.

#### Instructions for classification class:
 - All you really need is a clean dataset (without NaNs) split into your X (predictors) and y (outcome variable). From here you can assign your class to an object as below:
test_model = full_classification(X, y, baseline=0.606, shuffle=True, stratify=y, save_it=True)
 - As you can see here, some additional settings have been specified. For a list of the settings - see the docstring below. Note that by default the class will print out everything that it does, you need to specifically turn this setting off with print_info = False.
 - Once you have run - only the last model will be saved to the object, in order to reassign a specific model to the object you can re-run any one model using the functions below.
test_model.knn_model()
test_model.decision_tree_model()
test_model.logistic_model()
test_model.random_forest_model()
test_model.ADAboosting_model()
test_model.GradientBoosting()
test_model.NaiveBayes()
test_model.LinearSVC()
test_model.PolynomialSVC()
test_model.GaussianSVC()
test_model.MLP_Neural_Net()
- You will be able to see all of the settings that you are able to amend when you shift-tab on any of these, it is the standard scikit learn paramters. 
- The additional functions which have been included are:
    - test_model.knn_all_k() - this allows you to run a KNN model for a number of different Ks and will plot the accuracy curve over all of the Ks. the default is set to stop at K=50, but this can be changed in the settings.
    - test_model.decision_tree_model(print_tree=True, print_depth=3) - this allows you to not only run the decision tree, but also to graph out the tree. The default is to print to depth of 5.
    - test_model.NaiveBayes(power_transform=True) - it gives you the option to power transform the X data, Naive Bayes model works better with normally distributed data, and so this can help improve the model. 
    - test_model.coefs() - only works on models which have coefs to show (logistic), will print dataframe of the coefs.
    - test_model.gridsearch() - automatically runs a gridsearch on your current selected model. Returns model_grid model with best parameters. Has default parameters for each model type, but you can set your own by passing a dict into params = {}. When you print out the results before running gridsearch it will give you a good estimate of how long the gridsearch will take in minutes.
    - test_model.matrix_n_graphs() - will print out a pretty confusion matrix and if possible will print out a ROC curve and a precision recall curve for you to further be able to analyse your model. 
- The class includes an option to save every model which is run into a dataframe. This will be assigned to a global variable called model_tracker so that it can then be used outside of the dataframe. This is very useful in keeping track of all of your models, and keeps track of how long they take to run and all of the scores. 
- There is a demo file in the repo which shows what the class looks like in action on a spam dataset. 
Please feel free to reach out to me with any comments and ideas for improvements!

#### Docstring for the Classification class:
"""A class which automatically does all classification models and gridsearches for you (logisitic default). 
Note: when you run a new model it will overwrite the previous model. You can access the current model with .model and .model_des.
Other options:
   - run_all = default True, if set to false the class will not automatically run any models
   - baseline = default 0, set this to your baseline accuracy measure for comparision stats
   - standardize = default True, uses standard scaler and fit-transforms on train, transform on test if exists
   - test_size = default 0.15, decide the side of your test set - set to 0 if dont want test
   - folds = default 6, amount of folds for cross validation - integer > 1
   - shuffle = default True, shuffle the data for test split and cross val
   - stratify = default None, input the variable that you which to stratify the data by
   - print_info = default True, print all of the results out every time you run a model
   - save_it = default False, this adds functionality to be able to save down all model results into a dataframe, set as a global variable called model_tracker.
   - comment = default None, This is a comment field for the model_tracker
    Go to readme for further information: https://github.com/LukeBetham/machine-learning-classes/blob/master/README.md
    Created by LukeBetham"""

## Regression Class (work in progress) 
This class focuses on linear regression and is to be used for predicting continuous variables. It is still a work in progress and will be updated soon!


