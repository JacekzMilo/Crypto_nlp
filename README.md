# Aspect based semantic analysis with Python and NLP
The aim of this project was to create a program that will read cryptocurrency related articles and output the report with 
semantic analysis of sentences containing cryptocurrency names.   

**Latest update**: readme added.


## Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [To Do](#todo)
4. [File Descriptions](#descriptions)
5. [Steps](#steps)

## Installation
The code requires Python versions of 3.* and general libraries available through the Anaconda package. 
In addition, Scrapy, Pandas, NLTK, Googletrans, Numpy, Textblob, Seaborn, Re need to be 
installed to be able to run the script. 

## Project Motivation <a name="motivation"></a>
I wanted to create a bigger project that will extract data, analyse them and output result in a form of report. 
Through that project I wanted to learn as much as can in mentioned areas. 

## To Do <a name="todo"></a>
1. TODO: Add RSS trigger 
2. TODO: Upgrade reports
3. TODO: After some time, compare results from articles with real world crypto market changes.
4. TODO: Add more blogs
5. TODO: Upgrade code
6. TODO: Add analysis between sentences 
7. TODO: Add history of past articles
 
## File Description <a name="descriptions"></a>
In root folder there is a main.py file that runs all sub functions in proper order, to run this program, run main.py. 
Inside Scraper/Scraper/spiders/ there are spiders eg. zrozumiecbitc.py that scrap data from blogs. Every .py file 
refers to different blog. 
Inside Scraper/Scraper/ there is a custom_text_edit.py file that perform text cleaning and translation.

## Steps <a name="steps"></a>
# Gathering data and first cleanup

a) Gather data:

Scraping newest articles and saving them into .json file. Gathered data: article title, article text and article link

b) Clean initial data and translate:

After scrapping, every sentence has it's own quotes, custom_text_edit.py extract sentences and puts them into one 
string. Than ID's are assigned to every article. After that all articles that are in different language that English
are translated.

c) After converting initial data into proper format, code removes punctuation, special signs, urls, then lowercase but 
keeps "." at the end of each sentence. 

d) Remove stop words:

Stop words are those words that do not contribute to the deeper meaning of the phrase. They are the most common words 
such as: “the“, “a“, and “is“.

# Performing analysis

Once the texts have been carefully cleaned the next thing we did is, find the sentences that hold cryptocurrency coin 
names. After that a dataframe is created with article ID's, key sentences and their semantic analysis.

In the next step code performs sentence sentiment polarity in order to show the orientation of the expressed sentiment.

In the next steps code groups data into different dataframes that are loaded into GCP tables to be used in Data Studio
reports.

# Creating reports

Every article report consists of two pages. First page outputs article link, article title and bar plot with semantic
analysis of every sentence that contains cryptocurrency coin name. 

Second page showcase three plots that refer to semantic distribution. Top left graph is candle plot that show distribution
of sentence sentiment polarity. It consists of values: min, lower quartile, upper quartile and max. Top right plot is a
histogram that show the quantity of different coins grouped into different ranges. Bottom graph show semantic polarity
distribution grouped per coin, each bar refers to different sentence containing coin name.
Final report can be found under this link: https://datastudio.google.com/reporting/38998403-3617-4f5a-a8f0-7ac816c08a0d