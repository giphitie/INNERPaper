# Individualized Risk Assessment of Preoperative Opioid Use by Interpretable Neural Network Regression

## Overview

This project aims to understand the risk of preoperative opioid. A novel Interpretale Neural Network Regression (INNER) is proposed to conduct individualized risk assesment of preoperative opioid use. Intensive simulations and statistical analysis of 34,186 patients expecting surgery in the Analgesic Outcomes Study (AOS) show that the proposed INNER not only can accurately predict the preoperative opioid use based on preoperative characteristics as deep neural network (DNN), but also can estimate the patient-specific odds of opioid use without pain and the odds ratio of opioid use for one  unit increase in  the reported overall body pain, leading to more straightforward interpretations on opioid tendency compared to DNN.

## Reuqirements

The project uses Python 3.7.4 with ```Tensorflow == 2.0.0```, ```Scikit-learn == 0.20.2``` and ```Numpy == 1.17.4```.

## INNER Model
In the INNER model, we  utilize DNN to construct individualized coefficients in a logistic regression model, wherein  the regression coefficients are functions of individual characteristics. They lead to two metrics, Baseline Opioid Tendency (BOT) and Pain-induced Opioid Tendency (POT), which are useful for the individualized assessment of opioid use for each patient. In particular, BOT refers to the odds of receiving preoperative opioids when the patient does not report pain and POT is the odds ratio of  receiving preoperative opioids for one unit increase in the reported overall body pain. To print the summary of the INNER model architecture used for AOS data run:
```
$ python SummaryModel.py
```
<img align="middle" src="https://github.com/YumingSun/INNER/blob/master/utilities/ArchitecutreOfINNER.png">
