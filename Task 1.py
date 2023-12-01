#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk # Used for creating the GUI components.
from tkinter import ttk #Used for handling and processing tabular data.
import pandas as pd
import requests
from bs4 import BeautifulSoup # Used for web scraping to parse HTML content.
import time
import csv
import webbrowser #Used for opening web links.
from lxml import html #Used for processing HTML and XML.
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re #Used for making HTTP requests to fetch web pages.
import nltk #Natural Language Toolkit for text processing.


# In[2]:


#Retrieves publication links from a specified base URL using web scraping


# In[3]:


def Link_Retrieval(base_url, num_pages=9):
    Total_PL = []

    for page in range(0, num_pages + 1):
        Link_URL = f"{base_url}?page={page}"
        headers = {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        response = requests.get(Link_URL, headers=headers)

    # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to fetch {Link_URL}. Status: {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        links_on_page = [a["href"] for a in soup.find_all("a", href=True)]
        Total_PL.extend(links_on_page)

        # Polite crawling - add a delay between requests
        time.sleep(2)

    return Total_PL

if __name__ == "__main__":
    base_url = "https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-fluid-and-complex-systems-fcs/publications/"
    Total_PL = Link_Retrieval(base_url, num_pages=9)

    Publication_Link = [link for link in Total_PL if 'publications' in link and 'https://' in link]

    print(f"Total Number of Publication Links: {len(Publication_Link)}")
    for link in Publication_Link:
        print(link)


# In[4]:


# Collect details from links


# In[5]:


pd_data = []

for link_url in Publication_Link:
    response = requests.get(link_url)
    if response.status_code != 200:
        print(f"Failed to fetch {link_url}. Status code: {response.status_code}")
    else:
        tree = html.fromstring(response.content)
        xpaths = [
            '//p[@class="relations persons"]',
            '//div/h1/span',
            '//span[@class="date"][1]',
        ]

        details = []
        for xpath in xpaths:
            texts = tree.xpath(xpath)
            if texts:
                details.append(texts[0].text_content().strip())

        # Include the link in the details
        details.append(link_url)
        pd_data.append(details)


# In[6]:


#Performs text normalization, tokenization, removal of common stop words, and stemming.


# In[7]:


def clean_text(text):
    # Text normalization: Convert to lowercase
    text = text.lower()

    # Tokenization: Split text into words
    words = re.findall(r'\b\w+\b', text)

    # Remove common stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stemming: Reduce words to their base form
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

# Create DataFrame with columns for Authors, Title, Publication Date, and Link
dataframe1 = pd.DataFrame(pd_data, columns=['Authors', 'Title', 'Published Date', 'Link'])

# Assuming you have a function clean_text defined somewhere
# Apply clean_text to each column in the DataFrame
for column in ['Authors', 'Title', 'Published Date']:
    dataframe1[column] = dataframe1[column].apply(clean_text)

# Create DataFrame with columns for Authors, Title, Publication Date, and Link
 
dataframe1.to_csv('Published_Data.csv', index=False)
print("Data saved successfully")

# Processing data
processing_data = dataframe1.copy()


# In[8]:


#Creates an inverted index from the processed DataFrame.


# In[9]:


def create_inverted_index(dataframe):
    inverted_index = defaultdict(list)

    for index, row in dataframe.iterrows():
        document_id = index  # Assuming each row in the DataFrame represents a document

        # Process and tokenize text data (adjust as needed based on your data)
        authors = row['Authors'].split() if 'Authors' in row else []
        title = row['Title'].split() if 'Title' in row else []
        publication_date = row['Published Date'].split() if 'Published Date' in row else []

        # Create a list of unique terms from the document
        document_terms = set(authors + title + publication_date)

        # Update the inverted index
        for term in document_terms:
            inverted_index[term].append(document_id)

    return inverted_index


# In[10]:


# Build Inverted Index
inverted_index = create_inverted_index(processing_data)

# Example: Print the inverted index for the term 'spectral'
print("Inverted Index for 'spectral':", inverted_index.get('spectral', []))

# Sample processing for a single row
Sample = processing_data.loc[0, :].copy()
print("Sample Before:", Sample)

# Assuming you want to create a dictionary for indexing_trial
indexing_trial = {}

# Use the DataFrame index as the unique identifier
SerialNo = Sample.name
word = Sample.Title.split()[0]
example = {word: [SerialNo]}
indexing_trial.update(example)
print("Sample After:", Sample)
print("Indexing Trial:", indexing_trial)


# In[11]:


#Opens the link associated with a selected item in the Treeview (double-click event).
def open_link(event):
    item = tree.selection()[0]
    link = tree.item(item, 'values')[-1]
    if link:
        webbrowser.open(link)

#Loads data from a CSV file (Published_Data.csv).
def search_data(query):
    # Load the data
    dataframe = pd.read_csv("Published_Data.csv")

    # Filter the data based on the search query
    result_df = dataframe[
        dataframe.apply(lambda row: query.lower() in " ".join(row).lower(), axis=1)
    ]

    return result_df

#Clears previous search results in the Treeview.
def display_search_results(results):
    # Clear the previous search results
    for item in tree.get_children():
        tree.delete(item)

    # Insert new search results
    for index, row in results.iterrows():
        values = row.tolist()
        
        # Check if the row already exists in the Treeview
        existing_items = tree.get_children()
        is_duplicate = any(tree.item(item, 'values') == values for item in existing_items)
        
        if not is_duplicate:
            tree.insert("", "end", values=values)

#Retrieves the search query from the entry field and performs a search.
def search_button_clicked():
    query = entry.get()
    results = search_data(query)
    display_search_results(results)


# In[12]:


#GUI Creation


# In[13]:


# Create the main window
root = tk.Tk()
root.title("Publication Search Engine for Research Centre for Fluid and Complex Systems")
root.geometry("1000x800")

# Configure background color
root.configure(bg='#ADD8E6') # Light Blue color

# Create the search bar
label = tk.Label(root, text="Search:", font=("Helvetica", 16, "bold"), bg='#f0f0f0')
label.pack(side=tk.TOP, padx=10, pady=10)

entry = tk.Entry(root, width=40, font=("Helvetica", 14))
entry.pack(side=tk.TOP, padx=10, pady=5)

# Style the search button
style = ttk.Style()
style.configure("TButton", padding=6, relief="flat", background="#4299ff", foreground="black")
search_button = ttk.Button(root, text="Search", style="TButton", command=search_button_clicked)
search_button.pack(side=tk.TOP, padx=10, pady=5)


# In[ ]:


# Create the result table
frame = ttk.Frame(root)
frame.pack(padx=10, pady=10, side=tk.TOP, fill="both", expand=True)

tree = ttk.Treeview(
    frame,
    columns=("Authors", "Title", "Published Date", "Link"),
    show="headings",
    selectmode="browse",
)

# Configure header colors
style.configure("Treeview.Heading", background="#4299ff", foreground="black")

tree.column("Authors", width=200, anchor=tk.CENTER)
tree.column("Title", width=300, anchor=tk.W)
tree.column("Published Date", width=150, anchor=tk.CENTER)
tree.column("Link", width=200, anchor=tk.CENTER)

tree.heading("Authors", text="Authors")
tree.heading("Title", text="Title")
tree.heading("Published Date", text="Published Date")
tree.heading("Link", text="Link")

tree["height"] = 15

# Configure row colors
style.configure("Treeview", background="#f0f0f0", foreground="black")

vertical_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=vertical_scrollbar.set)
vertical_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
tree.pack(side=tk.LEFT, expand=True, fill="both")

# Clear the previous search results
for item in tree.get_children():
    tree.delete(item)

# Bind double-click event to open link
tree.bind("<Double-1>", open_link)

# Run the GUI
root.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




