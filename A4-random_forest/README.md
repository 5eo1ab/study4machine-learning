## Assignment 4 : Implement random forest

1. Build BaggingClassifier based on k-NN classifier

  >- You can use decision tree implemented by sklearn
  >- Set the number of bootstraped subsets, the size of subsets, the number of features to be sampled as parameters
  >  - Subsets should be created by bootstraping

Evaluation

  >- Use segmentation.data
  >- Split data into train and test set
  >    - [cross_validation.train_test_split](http://scikit-learn.org/stable/modules/cross_validation.html).
  >    - Test set size = 20% of original data
  >    - Use stratified split
  >    - Random state=100
  >- The number of estimators = [1, 5, 10, 20, 30, 50]
  >- The number of samples in each subset = 130
  >- The number of features = [10, 15, 19]
  >- min_samples_split = 10 for decision tree
  >- Calculate accuracy
  >    - Apply random forest algorithm for every pairs of (the number of estimators, the number of features)