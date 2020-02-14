# GFS

Application with necessary methods to repeat all the experiments from the article:

On the Interpretability of Machine Learning Models and Experimental Feature Selection in Case of Multicollinear Data

## Application

Application consists of a Python file `gfs.py` and other necessary Python modules. Required python packages are listed in the file `requirements.txt`.

## Menu functionality

- File:
  - Load dill: load a `.pkl` file containing a workspace snapshot
  - Save state: save current state of necessary variables to a `.pkl` file
  - Load state: load saved state of necessary variables from a `.pkl` file
  - Clear output: delete contents of main window (text output)
  - Exit: exit the application and close all windows
- Data:
  - Provide UCI-BCW: load raw data from the source of UCI-BCW dataset
  - Provide NSL-KDD: load raw data from the source of NSL-KDD dataset (data files should be in a subfolder with a name `NSL-KDD`)
  - Load last UCI-BCW: load saved state of necessary variables from a `UCI-BCW.pkl` file
  - Load last NSL-KDD: load saved state of necessary variables from a `NSL-KDD.pkl` file
- Descriptive:
  - Histogram: display histogram of the class values
  - Multicollinearity: display plot of absolute values of Spearman’s rank correlation coefficients of all distinct feature pairs for the dataset, ordered by their absolute value in descending order
  - Spearman heatmap: display plot of Spearman’s rank correlation coefficients as a heatmap (see function `rfpimp.plot_corr_heatmap`)
  - Feature dep. matrix: display plot of dependence of the features on other features (see function `rfpimp.feature_dependence_matrix`)
- Base learning:
  - Learn: train a classifier from a whole dataset
  - Importance: display feature importances obtained from the trained classifier
  - Predict: run a prediction on the trained classifier using a whole dataset
  - Permut. imp.: calculate permutation feature importance using the method `rfpimp.importances` using a whole dataset
  - Drop-column imp.: calculate drop-column feature importance using the method `rfpimp.dropcol_importances` using a whole dataset
- Greedy FS:
  - Calculate: run the greedy feature selection algorithm to obtain the reduced dataset
  - Predict: run a prediction on the trained classifier using the reduced dataset
  - Cumulative feature imp.: display accuracy depending on the features, ordered descendingly by accuracy, for each step
  - Accuracy growth: display a plot of accuracy depending on steps
  - Show growth stat.: display boxplots of accuracy depending on steps from all the experiments
  - Permut. imp.: calculate permutation feature importance using the method `rfpimp.importances` using the reduced dataset
  - Drop-column imp.: calculate drop-column feature importance using the method `rfpimp.dropcol_importances` using the reduced dataset
