# Sentiment-Analysis
Developed sentiment analysis system for customer reviews using NLP and ML techniques. Accurately classifies sentiments and clusters reviews based on aspects. Valuable insights for businesses' decision-making in data-driven landscape.

#Introduction
This project aims to develop a sentiment analysis system for customer reviews, utilizing 
natural language processing and machine learning techniques. The dataset consists of 
customer reviews and associated labels, such as positive, negative, or neutral sentiments. The 
machine learning model is trained on 70% of the data and tested on the remaining 30%, with 
the input text coming from the "Review" and "Summary" columns.
The aim of the project was to create a system that can accurately classify the sentiment of 
customer reviews, providing businesses with valuable insights into the opinions of their 
customers. Additionally, the project involves the clustering of reviews based on the aspects 
that are linked to specific sentiments, such as identifying the top reasons for negative or 
positive reviews for a given product. Overall, this project is a crucial tool for businesses in 
today's data-driven landscape, allowing them to make informed decisions about product 
improvements and marketing strategies based on customer feedback.

<b>Data Collection and Pre-processing</b>
<newline>
Dataset was downloaded from Kaggle. The link of the dataset has been given below. It was 
accessed on 8th April 2023.
https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset
For pre-processing, the unwanted column such as the product_name, Rate, product_price was 
removed. All Nonvalues were removed. The Sentiment column of the dataset was mapped on 
the sentiment_map = {‘positive’:1,’neutral’:0, ‘negative’: -1}
Size of the data (before pre-processing) = 205052*6
Size of the data (after pre-processing) = 18370*5

###Distribution of the Sentiment(labels)
![image](https://github.com/acmax406/Sentiment-Analysis/assets/79563144/d3faab6a-718d-46c0-9ada-3e24d1e6c283)


##Distribution of the Ratings
![image](https://github.com/acmax406/Sentiment-Analysis/assets/79563144/24d431e3-e32a-4f19-96fb-cfabc79816b7)

#Methodology
##BERT architecture 
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained deep 
learning model for natural language processing. It uses a transformer-based architecture that 
allows it to process words in a bidirectional manner, enabling it to better understand the 
context of a sentence. BERT has been shown to be highly effective for sentiment analysis 
tasks due to its ability to capture the subtle nuances of language and its pre-training on large 
amounts of text data. By fine-tuning BERT on sentiment analysis datasets, it can accurately 
predict the sentiment of a given piece of text.
#Pre-Training Process
Firstly, the code reads the dataset and pre-processes it by dropping irrelevant columns, 
mapping the sentiment values to numbers, and splitting the data into training, validation, and 
testing sets. It also imports the required libraries, including the DistilBertTokenizerFast and 
TFDistilBertModel from the Transformers library.
Secondly, the code defines a function called batch_encode to encode text data in batches 
using the DistilBERT tokenizer.
Thirdly, the code configures the DistilBERT initialization, makes its layers untrainable, and 
sets some hyperparameters such as the maximum length of the input sequence, layer dropout, 
learning rate, and random seed.
Finally, the code defines a function called build_model to build the neural network model. It 
takes the pre-trained DistilBERT as an input, and it defines the layers and the number of 
neurons for each layer. It also uses dropout regularization to prevent overfitting. The model 
uses SoftMax activation function.
After that we tried to do the topic modelling on a given dataset of reviews using Latent 
Dirichlet Allocation (LDA) and assigns topics to each review. The code reads the dataset 
from a CSV file and pre-processes the text data by removing stop words, making bigrams and 
trigrams, and lemmatizing the text. It then creates a dictionary and a corpus and trains an 
LDA model on the corpus. Finally, the code assigns a topic to each review based on the most 
probable topic from the LDA model.
The generated graph insights into the topics discussed in the reviews and their corresponding 
sentiment, providing an understanding of which topics are generally positive or negative.
#Results
The accuracy of the model was approximately 90.47% with the validation accuracy of 
91.48%.
The results of the clustering the summaries according to the aspects based on which the 
sentiment have been assigned
![image](https://github.com/acmax406/Sentiment-Analysis/assets/79563144/ed0f4d84-4f20-411c-87a1-bc8fac1d53ef)

Project aim was to perform sentiment analysis and topic modelling on a dataset of reviews 
and comments from an e-commerce website. The results suggest that the model was able to 
accurately classify the sentiment of the reviews, with an overall accuracy of approximately 
90.47%. The validation accuracy of 91.48% further supports the model's effectiveness.
The use of Latent Dirichlet Allocation (LDA) for topic modelling allowed for a deeper 
understanding of the topics discussed in the reviews and their corresponding sentiment. The 
generated graph provides insights into which topics are generally positive, negative, or 
neutral. This information can be useful for businesses to understand customer preferences and 
identify areas for improvement.
The results of the clustering of the summaries based on the aspects of the reviews and their 
assigned sentiments suggest that the majority of reviews were positive across all topics. 
However, some topics had a higher proportion of negative reviews compared to others. For 
example, topic 1 had a higher proportion of negative reviews than other topics, while topics 7 
and 8 were predominantly positive.
Overall, we can further improve the model accuracy using the method such as 
oversampling/under sampling as we look at the data, it is highly imbalanced. This technique 
involves creating additional samples in the minority class (i.e., the class with fewer samples) 
to balance the dataset. This can be achieved by duplicating existing samples or generating 
new ones using techniques like SMOTE (Synthetic Minority Over-sampling Technique). The 
goal is to increase the representation of the minority class in the dataset and help the model 
learn to better distinguish between classes. While the under sampling involves reducing the 
number of samples in the majority class (i.e., the class with more samples) to balance the 
dataset. This can be done by randomly selecting a subset of samples from the majority class. 
The goal is to reduce the bias towards the majority class and encourage the model to learn 
from the minority class as well. Anyways, the methodology used in this project appears to be 
effective in analysing sentiment and identifying topics in a dataset of customer reviews. The 
insights gained from this analysis can be useful for businesses to improve their products or 
services and enhance customer satisfaction.
