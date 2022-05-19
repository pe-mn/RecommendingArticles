# Recommending Web Articles

## DataSet
JSON file with 3 columns (body, title, category)
Each article falls into one of three categories: 
- Engineering, 
- Product & Design,
- Startups and Business

## THOUGHT PROCESS
- In order to extract features from the text we can use (Bag of Words, N_grams or TF-IDF). And to cluster our categories into subcategories, we can either use the title, or the article body. So, I examined combinations of the above.
- I could reach a classifier accuracy of 85% using TF-IDF on the body, and using the XGBoost Classifier.
- For the Subcategory Clusters, I got different results, but saved those created using BOW on the titles, as I believe titles are more intuitive and representative in clustering, and I think BOW works best with titles than the TF-IDF method.

## Data Cleaning
- Check for NaNs
- Check for Duplicates
- Text Preprocessing

## Supervised Part 
Built a Supervised Learning Model (Classifier) and achieved an accuracy of 85%,
using the full article body.

## Unsupervised Part
Clustered each of the 3 category to SubCategories