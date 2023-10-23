# stm_app.py
Data Info:
Problem Statement

We have a advertising dataset of a marketing agency. Goal is to develop a ML algorithm that predicts if a particular user will click on an advertisement. The dataset has 10 features:

'Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country', Timestamp' 'Clicked on Ad'.

'Clicked on Ad' is the categorical target feature, which has two possible values: 0 (user didn't click) and 1(user clicked).

In this model we are going to assign values to ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male'] features to get prediction wehter user is going to click on Ad or not depending on these features values.

We use Logistic Regression Model and this model shows Accuracy Score of 98% asosiated with recall score 99% and precision score 97% during testing.
