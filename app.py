import pickle
import pandas as pd
import streamlit as st
import sklearn
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Delhi Capitals',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals'
]

cities = ['Mohali', 'Cuttack', 'Mumbai', 'Bangalore', 'Hyderabad', 'Chennai',
          'Pune', 'Raipur', 'Bengaluru', 'Johannesburg', 'Chandigarh',
          'Port Elizabeth', 'Kolkata', 'East London', 'Jaipur', 'Delhi',
          'Ahmedabad', 'Bloemfontein', 'Durban', 'Cape Town', 'Sharjah',
          'Centurion', 'Dharamsala', 'Abu Dhabi', 'Ranchi', 'Indore',
          'Visakhapatnam', 'Kimberley', 'Nagpur']

st.header('IPL Win Predictor')

pipe = pickle.load(open('pipe.pkl', 'rb'))

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team', teams)
with col2:
    bowling_team = st.selectbox('Select Bowling Team', teams)

city = st.selectbox('Select Host city', cities)

target = st.number_input("Target")

col3, col4, col5 = st.columns(3)

with col3:
    Runs = st.number_input('Runs')
with col4:
    Overs = st.number_input('Overs Completed')
with col5:
    wicket = st.number_input('Wickets')

if st.button('Predict'):
    Runs_Left = target - Runs
    Balls_Left = 120 - (Overs * 6)
    wickets_left = 10 - wicket
    CRR = Runs / Overs
    RRR = Runs_Left * 6 / Balls_Left
    DF = pd.DataFrame(
        {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [city], 'Runs_Left': [Runs_Left],
         'Balls_Left': [Balls_Left], 'wickets_Left': [wickets_left], 'total_runs_x': [target], 'CRR': [CRR],
         'RRR': [RRR]})
    result = pipe.predict_proba(DF)
    loss = str(round(result[0][0], 4) * 100)
    win = str(round(result[0][1], 4) * 100)
    st.text(batting_team + " Win Probability is  " + win + " %")
    st.text(bowling_team + " Win Probability is  " + loss + " %")
