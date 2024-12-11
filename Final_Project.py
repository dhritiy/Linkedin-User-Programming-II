import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st

st.markdown("# Linkedin Predictor <img src = 'https://similarpng.com/linkedin-logo-transparent-png' width = '50'/>")
st.markdown("### Predict whether someone is a Linkedin user and the probability of them using Linkedin based on their background:")

# Rest of your code remains the same...
def data_and_model():
    s = pd.read_csv("social_media_usage.csv")

    def clean_sm(x):
        return np.where(x == 1, 1, 0)

    ss = pd.DataFrame({
        'sm_li': clean_sm(s['web1h']),
        'income': np.where(s['income'] <= 9, s['income'], np.nan),
        'education': np.where(s['educ2'] <= 8, s['educ2'], np.nan),
        'parent': clean_sm(s['par']),
        'married': clean_sm(s['marital']),
        'female': clean_sm(s['gender']),
        'age': np.where(s['age'] <= 98, s['age'], np.nan)
    })

    ss = ss.dropna()

    y = ss['sm_li']
    X = ss[['income', 'education', 'parent', 'married', 'female', 'age']]

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        stratify = y,  
                                                        test_size = 0.2, 
                                                        random_state = 58) 

    lr = LogisticRegression(class_weight = 'balanced')

    lr.fit(X_train, y_train)

    return ss, lr

ss, lr = data_and_model()
    
age = st.number_input('Age', min_value = 18, max_value = 98, value = 30,
                    help='Select age between 18 and 98')
    
income = st.select_slider('Household Income', 
                            options = list(range(1, 10)),
                            value = 5,
                            format_func = lambda x: {
                                1: "Less than $10,000",
                                2: "10 to $19,999",
                                3: "$20 to $29,999",
                                4: "$30 to $39,999",
                                5: "$40 to $49,999",
                                6: "$50 to $74,999",
                                7: "$75 to $99,999",
                                8: "$100 to $149,999",
                                9: "$150,000 or more"
                            }[x])

    
education = st.select_slider('Education Level',
                               options = list(range(1, 9)),
                               value = 4,
                               format_func = lambda x: {
                                   1: "Less than high school",
                                   2: "High school incomplete",
                                   3: "High school graduate",
                                   4: "Some college, no degree",
                                   5: "Two-year associate degree",
                                   6: "Four-year college degree",
                                   7: "Some postgraduate schooling",
                                   8: "Postgraduate/Professional degree"
                               }[x])

female = st.radio('Gender', ['Male', 'Female'], index = 0,
                     horizontal = True)
female = 1 if female == 'Female' else 0

parent = st.radio('Parent Status', ['Not a Parent', 'Parent'], 
                     index = 0, horizontal=True)
parent = 1 if parent == 'Parent' else 0


married = st.radio('Marital Status', ['Not Married', 'Married'], 
                      index =0, horizontal=True)
married = 1 if married == 'Married' else 0



# Create prediction button with better styling
if st.button('Predict LinkedIn Usage', use_container_width=True):
    # Create input data frame
    input_data = pd.DataFrame({
        'income': [income],
        'education': [education],
        'parent': [parent],
        'married': [married],
        'female': [female],
        'age': [age]
    })
    

    pred = lr.predict(input_data)[0]
    prob = lr.predict_proba(input_data)[0][1]
    
    st.subheader('Prediction Results')
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if pred == 1:
            st.success('LinkedIn User')
        else:
            st.error('Not a LinkedIn User')
    
    with res_col2:
        st.metric(
            label="Probability",
            value=f"{prob:.1%}"
        )
    



