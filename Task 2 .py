#!/usr/bin/env python
# coding: utf-8

# #Import Libriers 

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics


# #Data Loading and Exploration

# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
data_frame = pd.concat([train,test],axis = 0)


# #This part loads training and testing data from CSV files, concatenates them into a single DataFrame df.

# In[3]:


data_frame.head(10)


# In[4]:


# Defines lenght of dataset
len(data_frame)


# In[5]:


# provides data info
data_frame.info()


# In[6]:


# provides shape of dataset - 3 attribites and 127600 entity.
data_frame.shape


# In[7]:


data_frame.describe()


# In[8]:


# checking the unique values of Class Index
data_frame['Class Index'].unique() 


# #Data Preprocessing:

# In[9]:


# Convert numerical labels to corresponding categories
data_frame['Class Index'].replace([1, 2, 3, 4], ['World', 'Sports', 'Business', 'Sci/Tech'], inplace=True)


# In[10]:


# checking the unique values of Class Index after converting to categories.
data_frame['Class Index'].unique() 


# In[11]:


data_frame.describe()


# In[12]:


data_frame.head()


# In[13]:


#Exploratory Data Analysis (EDA)


# In[14]:


# Plot a pie chart for the distribution of categories
plt.figure(figsize=(8, 8))
data_frame['Class Index'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
plt.title('Distribution of Categories in Class Index')
plt.show()


# In[15]:


# Count the occurrences of each category in 'Class Index'
category_counts = data_frame['Class Index'].groupby(data_frame['Class Index']).count()
print(category_counts)


# In[16]:


# checking for any null values
data_frame.isnull().sum()


# #Text Preprocessing: Tokenizes, lemmatizes, and stems the text in 'Description' and 'Title' columns, then combines them into a single 'Text' column.

# In[17]:


# Tokenization, Lemmatization, and Stemming functions
def tokenize(text):
    return word_tokenize(text)

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def stem(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

# Apply tokenization, lemmatization, and stemming to 'Description' column
data_frame['Description'] = data_frame['Description'].apply(tokenize)
data_frame['Description'] = data_frame['Description'].apply(lemmatize)
data_frame['Description'] = data_frame['Description'].apply(stem)

# Apply tokenization, lemmatization, and stemming to 'Title' column
data_frame['Title'] = data_frame['Title'].apply(tokenize)
data_frame['Title'] = data_frame['Title'].apply(lemmatize)
data_frame['Title'] = data_frame['Title'].apply(stem)

# Example: Display processed 'Description' for the first few rows
print("Processed 'Description' for the first few rows:")
print(data_frame['Description'].head())

# Example: Display processed 'Title' for the first few rows
print("\nProcessed 'Title' for the first few rows:")
print(data_frame['Title'].head())

# Combining 'Title' and 'Description' into a single column
data_frame['Text'] = data_frame['Title'].apply(lambda x: ' '.join(x)) + ' ' + data_frame['Description'].apply(lambda x: ' '.join(x))

# Example: Display the combined 'Text' for the first few rows
print("\nCombined 'Text' for the first few rows:")
print(data_frame['Text'].head())


# In[18]:


data_frame.head()


# # Train-Test Split:

# In[19]:


# Assuming 'Text' is the column containing the processed text data, and 'Class Index' is the target variable
X = data_frame['Text']
y = data_frame['Class Index']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# In[20]:


X.head()


# In[21]:


y.head()


# # Text Vectorization and Model Training:
# Uses TF-IDF vectorization to convert text data into numerical format and trains a Multinomial Naive Bayes classifier.

# In[22]:


# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# SVM classifier #Training time can be longer, sensitivity to choice of kernel and hyperparameters

# In[ ]:


# SVM Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)


# In[ ]:


# Multinomial Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)


# In[ ]:


# Make predictions
y_pred_svm = svm_classifier.predict(X_test_tfidf)
y_pred_nb = nb_classifier.predict(X_test_tfidf)


# # Model Evaluation

# In[ ]:


# Evaluate SVM Classifier
print("SVM Classifier:")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred_svm))
cm = confusion_matrix(y_test, y_pred_svm)
#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            fmt='g',cmap="flare")
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()


# In[ ]:


# Evaluate Multinomial Naive Bayes Classifier
print("\nMultinomial Naive Bayes Classifier:")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", metrics.classification_report(y_test, y_pred_nb))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred_nb))

cm1 = confusion_matrix(y_test, y_pred_nb)
#Plot the confusion matrix.
sns.heatmap(cm1,
            annot=True,
            fmt='g',cmap="flare")
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()


# # Text Classification Function:Defines a function classify_text to predict the category of new input text.

# In[ ]:


# Function to classify new input text
def classify_text(text, model):
    # Convert the input text to TF-IDF vector
    text_tfidf = vectorizer.transform([text])

    # Classify the text using the specified model
    prediction = model.predict(text_tfidf)

    # Return the predicted category
    return prediction[0]


# # Example Usage: Takes user input and predicts the category using the trained model.

# In[ ]:


# user input
user_input = input("Enter a AG news article: ")


# In[ ]:


# Example usage with Multinomial Naive Bayes
user_prediction_nb = classify_text(user_input, nb_classifier)
print(f"The predicted category for the input using Naive Bayes is: {user_prediction_nb}")


# In[ ]:


# Example usage with SVM
user_prediction_svm = classify_text(user_input, svm_classifier)
print(f"The predicted category for the input using SVM is: {user_prediction_svm}")


# In[ ]:





# In[ ]:





# In[ ]:




