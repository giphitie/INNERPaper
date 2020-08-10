# Demo

## Data
The training dataset, `TrainingData.csv`, and the testing dataset, `TestingData.csv`, are simulated as described in the _Simulation Study_ part of the papaer. There are 16,000 observations in the training dataset and 4,000 observations in the testing dataset. Both of the two datasets have one label and three features. 

## Scripts
* `train.py`: The script used to train the INNER model. To train the INNER model run:
```
$ python train.py PATH_TO_TRAINING_DATA PATH_TO_MODEL
```
It trains the INNER model using training data at `PATH_TO_TRAINING_DATA` and saves the trained model at `PATH_TO_MODEL`. Pre-trained model is `pre_trained_model.h5`.

* `test.py`: The script used to evaluate the performance of the INNER model. To evaluate the performance run:
```
python test.py PATH_TO_TESTING_DATA PATH_TO_MODEL
```
It evaluate the performance of pre-trained model at `PATH_TO_MODEL` using testing data at `PATH_TO_TESTING_DATA`. It prints out the C statistics, accuaracy, sensitivity, specificity and balance accuracy.

*`estimate.py`: The script used to estimate the BOT and POT. To estimate these two metrics run:
```
python estimate.py PATH_TO_ESTIMATE_DATA PATH_TO_MODEL PATH_TO_OUTPUT
```
It estimate the BOT and POT for each subject in the data at `PATH_TO_ESTIMATE_DATA` using pre-trained model at `PATH_TO_MODEL`. It prints out the first five results and save all the results at `PATH_TO_OUTPUT`.
