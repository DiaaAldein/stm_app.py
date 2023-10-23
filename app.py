import pandas as pd
import streamlit as st
import joblib

df = pd.read_csv("advertising.csv")

st.title("Click on Ads advertising ML Model")

st.header('Data Info: ')
st.info('''
Project prepared by: Diaa Aldein Alsayed Ibrahim Osman\n
Prepared for: Epsilon AI Institute\n
Problem Statement:\n
We have a advertising dataset of a marketing agency. Goal is to develop a ML algorithm that predicts if a particular user will click on an advertisement. The dataset has 10 features:
\n
'Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Ad Topic Line', 'City', 'Male', 'Country', Timestamp' 'Clicked on Ad'.
\n
'Clicked on Ad' is the categorical target feature, which has two possible values: 0 (user didn't click) and 1(user clicked).
\n
In this model we are going to assign values to ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male'] features to get prediction
wehter user is going to click on Ad or not depending on these features values.
\n
We use Logistic Regression Model and this model shows Accuracy Score of 98% asosiated with recall score 99% and precision score 97% during testing.''')

st.subheader('The below table show statistical description for datasets used for this Model it is useful for user to check it to indicate the range of input features in oreder to get almost 99% accuracy in predicted result: ')
st.dataframe(df.describe())

st.subheader('Some Insights on dataset from Data Analysis')
st.write('''
         1. Daily Time Spent on Site: the more time spent by customers on site they tend less to clicking on Ad and vice versa.
2. Age: the younger the age of customer visiting the site the less they are clicking on ad and vise versa.
3. Area Income: the higher income the less clicking on ad and vis versa.
4. Daily Internet Usage: the more user spent time on the internet they are tend to not clicking on add and vice versa.
5. Male: there no clear effect of gender on clicking on Ad between male or female.
         ''')

with st.sidebar:
    st.header('User Input Features For the Model: ')

    daily_time_spent_on_site = st.slider('1. Insert Daily Time Spent on Site in minute:',0,100,33)
    st.write('The time you insert is ',daily_time_spent_on_site,'m remeber the model is trained in range from (32.6 m to 91.43 m) for 98% accuracy')

    age = st.slider('2. Insert user Age in years:',0,130,19)
    st.write('The user age inserted is ',age,'years old remeber the model is trained in range from (19 years to 61 years) for 98% accuracy')

    area_income = st.slider('3. Insert Area Income for user in $:', 10000,100000,14000)
    st.write('The Area Income you insert is ',area_income,' remeber the model is trained in range from (13,996.5 to 79,484.8) for 98% accuracy')

    daily_internet_usage = st.slider('4. Insert User Daily Internet Usage in mb:',100,300,105)
    st.write('The User Daily Internet Usage you insert is ',daily_internet_usage,'mb remeber the model is trained in range from (104.78 mb to 269.96 mb) for 98% accuracy')

    male = st.selectbox('5. Is user is Male ? select 1 for True & 0 For Faluse :',[0,1])
    st.write('The gender you select is ',male)

    data = [daily_time_spent_on_site,age,area_income,daily_internet_usage,male]

    scaler = joblib.load('scaler.h5')
    data_scaled = scaler.transform([data])

    model = joblib.load('model_lr.h5')
    result = model.predict(data_scaled)

    def show(result):
       if result == 1:
          return 'User will click on Ad'
       else:
          return 'User Will Not click on Ad'

    st.subheader('''Base on the above value entered The Model Prediction is:\n 1 for user is going to click on Ad 
    \n And, 0 for not going to click on Ad ''')

    if st.button('Predict'):
      st.write(show(result))

