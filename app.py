import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
news_df = pd.read_csv("train.csv")
news_df = news_df.fillna(" ")
news_df["content"] = news_df["author"] + " " + news_df["title"]
X = news_df.drop("label", axis=1)
y = news_df["label"]

# Define stemming function
ps = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub("[^a-zA-Z]", " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        ps.stem(word)
        for word in stemmed_content
        if not word in stopwords.words("english")
    ]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content


# Apply stemming function to content column
news_df["content"] = news_df["content"].apply(stemming)

# Vectorize data
X = news_df["content"].values
y = news_df["label"].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

# Styling and layout
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f7, #e0e7ff);
        padding: 3rem;
    }
    .icon {
        font-size: 3rem;
        color: #3498db;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .title {
        font-family: 'Montserrat', sans-serif;
        color: #2c3e50;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 4px;
        text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
        background: linear-gradient(to right, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .progress-container {
        position: relative;
        width: 100%;
        text-align: center;
        margin-top: 2rem;
    }
    .progress-text {
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        font-size: 1.2rem;
        font-weight: bold;
        color: #3498db;
    }
    .input-label {
        font-size: 1.4rem;
        font-family: 'Roboto', sans-serif;
        color: #e74c3c;  /* เปลี่ยนสีของข้อความที่นี่ */
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: block;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2); 
    }
    .input-text {
        margin-bottom: 1.5rem;
        width: 100%;
        padding: 12px;
        border: 2px solid #3498db;
        border-radius: 8px;
        font-size: 1.1rem;
        font-family: 'Roboto', sans-serif;
        background-color: #f0f4f7;
        color: #34495e;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        transition: border-color 0.3s ease;
    }
    .input-text:focus {
        border-color: #2980b9;
        outline: none;
    }
    .submit-btn {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 8px;
        font-size: 1.3rem;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease;
        display: block;
        margin: 0 auto;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    .submit-btn:hover {
        background-color: #2980b9;
        transform: scale(1.05);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
    }
    .result {
        margin-top: 30px;
        font-size: 1.7rem;
        text-align: center;
        color: #34495e;
        opacity: 0;
        animation: fadeIn 1.5s forwards;
    }
    .highlight {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3498db;
    }
    @keyframes fadeIn {
        to {
            opacity: 1;
        }
    }
    .result span {
        font-weight: bold;
        padding: 0 5px;
    }
    .result span.real {
        color: #2ecc71;
    }
    .result span.fake {
        color: #e74c3c;
    }
    .footer {
        font-size: 0.9rem;
        text-align: center;
        margin-top: 50px;
        color: #95a5a6;
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Icon and title of the web app
st.markdown('<div class="icon">📰</div>', unsafe_allow_html=True)
st.markdown('<div class="title">Fake News Detector</div>', unsafe_allow_html=True)

# Convert accuracy to percentage for display
accuracy_percentage = accuracy * 100

# แสดงความแม่นยำของโมเดลในรูปแบบ progress bar
st.markdown(
    '<div class="highlight">Model Accuracy <span>📊</span></div>',
    unsafe_allow_html=True,
)
st.progress(accuracy)  # Display progress bar (accuracy is a value between 0 and 1)
st.markdown(
    f"""
    <div class="progress-container">
        <div class="progress-text">{accuracy_percentage:.2f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input text field with styled label
st.markdown(
    '<label class="input-label">Enter news Article <span>📝</span></label>',
    unsafe_allow_html=True,
)
input_text = st.text_input(
    "",
    placeholder="Type your news article here...",
    key="input",
    help="Please input your news article to analyze.",
)

# Submit button for prediction
submit_button = st.button("Submit", key="submit")


# Function to predict fake or real news
def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]


# Handle button click for prediction
if submit_button:
    if input_text:
        pred = prediction(input_text)
        if pred == 1:
            st.markdown(
                '<div class="result">The News is <span class="fake">Fake</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result">The News is <span class="real">Real</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="result">Please enter a news article to analyze</div>',
            unsafe_allow_html=True,
        )

# Footer
st.markdown(
    '<div class="footer">Powered by Streamlit | Fake News Detector</div>',
    unsafe_allow_html=True,
)
