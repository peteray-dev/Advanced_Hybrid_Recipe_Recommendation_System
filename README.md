# Recipe_Recommendation_App

# A HYBRID RECOMMENDATION SYSTEM: ENHANCING OPTIMAL ALGORITHMIC PERFORMANCE (Food Recipes Recommendation System)

# INTRODUCTION

Over the years, recommendation systems have become a key tool for various businesses and organizations, enhancing user satisfaction and revenue growth. Many businesses that have integrated these systems have experienced significant increases in revenue. Recognizing the potential of recommendation systems in the food sector, this project aims to develop a system optimized for recommending recipes to users who use the recipe applications. Recipes will be suggested based on factors such as ratings, descriptions, keywords, and categories.
To better understand traditional recommendation processes used across various sectors and applications, this study explores several techniques, including collaborative filtering, content- based filtering, and hybrid methods. Content-based filtering utilizes Natural Language Processing (NLP) techniques to enhance recommendations by removing stop words from recipe descriptions and comparing similarities between vectors to suggest recipes that closely align with the user's past interactions. In contrast, collaborative filtering recommends recipes based on similar users preferences, using the K-Nearest Neighbors (KNN) algorithm to calculate similarity and identify neighboring recipes with shorter distances.
Recently, advanced techniques such as Graph Neural Networks (GNNs) have shown promise in personalized recommendation systems, particularly in e-commerce. Integrating GNNs into recipe recommendation systems could significantly enhance user experience by better aligning recommendations with individual preferences. The GNNs have the ability capture intricate relationships that exist between recipes that traditional methods might miss. This study employs GNNs with link regression to predict user ratings and recommend recipes, addressing challenges such as data sparsity as experienced in collaborative filtering and the cold start problem. This approach allows for more accurate rating predictions for new items on the platform and recommendations, thereby improving user satisfaction and engagement.

# PROBLEM STATEMENT

The widespread of the use of recommendation system in various areas which create a personalized experience for its user, it is expedient that the food industry also experience a transformed recommendation that suites the user’s expectation particularly the recipe applications. Traditional methods of recommendation such as collaborative and content-based techniques have shown an
 
impact in recommending items to user but fails in effectively addressing some issues like data sparsity, cold start problem, and capturing complex relationship. Collaborative is faced with sparse user item interaction while content-based fail to capture the relationship that exist between user and item.
Recently, techniques such as Graph Neural Networks (GNN) has created an improved recommendation system by exploring complex relationships, they have as well shown potential in handling data sparsity and cold start problem by leveraging on the graph structure of user-item interactions to make an accurate prediction and recommendation. However, integrating GNN into recipe recommendation is still underexplored and then requires further study to enhance user’s experience.
Therefore, this study will explore the traditional techniques to understand its approach, improve its usages and then also incorporate the GNN models such as GraphSAGE and GAT to optimize and personalize recipe suggestions. The goal is to capture intricate and complex relationships that exist between the user and recipe, overcome the problem faced by the traditional methods and then improve user experience.

# METHODOLOGY
the process of creating a personalized food recipe recommendation system designed to enhance user experiences is explored. In today's digital age, recommendation systems play a crucial role in increasing user engagement and revenue, as evidenced by platforms like Amazon. Beyond e-commerce, these systems are also making significant impacts in sectors such as education, tourism, and employment, influencing users' decisions and behaviors.
This project focuses on developing a food recipe recommendation system aimed at helping users improve their culinary skills and discover easy and quick dishes. By tailoring recipe suggestions to individual preferences, this system can be integrated into food recipe platforms, significantly enhancing the user experience. Whether users are looking to expand their cooking repertoire or find convenient meal options, their needs are intended to be met effectively and efficiently.
Traditional recommendation systems typically rely on methods such as collaborative filtering and content-based filtering. Collaborative filtering makes recommendations based on the preferences of similar users, while content-based filtering recommends items similar to those a user has liked in the past. While effective, these methods often overlook the complex and multifaceted nature of user preferences and item attributes.
To create a more sophisticated and accurate recommendation system, traditional methods were explored, and a graph neural network (GNN) approach was implemented. Unlike conventional systems that primarily focus on user-item interactions, GNNs consider the intricate relationships and dependencies within the data. This advanced method allows for a deeper understanding of user preferences and item characteristics, leading to more precise and personalized recommendations.
 
![image](https://github.com/user-attachments/assets/366e8ec9-2d6f-4bf8-ab16-dc38e5f1f19d)

The recipe data, which contains the recipe information, comprises 28 features, including: recipeid, name, authorid, authorname, cooktime, preptime, totaltime, datepublished, description, images, recipecategory, keywords, recipeingredientquantities, recipeingredientparts, aggregatedrating, reviewcount, calories, fatcontent, saturatedfatcontent, cholesterolcontent, sodiumcontent, carbohydratecontent, fibercontent, sugarcontent, proteincontent, recipeservings, recipeyield, and
 
recipeinstructions. The review dataset contains eight features: reviewid, recipeid, authorid, authorname, rating, review, datesubmitted, and datemodified. The features are of different datatypes, such as integer, object, and float. Below is an in-depth description of the features: Recipe Dataset:

Recipeid: The unique identifier for each recipe.

Name: The name of the recipe.

Authorid: The unique identifier of the user.

Authorname: The name of the user.

Cooktime: The time required to cook the recipe.

Preptime: The time required to prepare the recipe.

Totaltime: The total time required to make the recipe (i.e., the sum of cooktime and preptime).

Datepublished: The date the recipe was uploaded and published on the website.

Description: Brief information about the recipe.

Images: Links to images of the recipe.

Recipecategory: The category the recipe belongs to (e.g., frozen desserts).

Keywords: The main words associated with the recipe.

Recipeingredientquantities: The quantity of each ingredient used to make the recipe.

Recipeingredientparts: The names of the ingredients required to make the recipe. 

Aggregatedrating: The average rating of the recipe (i.e., the sum of the ratings divided by the total number of times the recipe was rated).

Reviewcount: The total number of reviews for the recipe.

Calories: The caloric content of the recipe.

Fatcontent: The fat content in the recipe. Saturatedfatcontent: The saturated fat content in the recipe. Cholesterolcontent: The cholesterol content in the recipe.

Sodiumcontent: The sodium content in the recipe. Carbohydratecontent: The carbohydrate content in the recipe. Fibercontent: The fiber content in the recipe.

Sugarcontent: The sugar content in the recipe. Proteincontent: The protein content in the recipe. Recipeservings: The number of servings the recipe yields.
 
Recipeyield: The yield of the recipe (e.g., number of pieces). Recipeinstructions: Step-by-step instructions for preparing the recipe. 

Review Dataset:

This dataset consists primarily of information about user interactions with the recipes. The features include:

Reviewid: The unique identifier for each review made by a user.

Recipeid: The unique identifier for the associated recipe. 

Authorid: The unique identifier for the user who made the review. 

Authorname: The name of the user who submitted the review.

Rating: The rating the user gave to a recipe, ranging from 1 to 5.

Review: The user's comments about the recipe. 

Datesubmitted: The date when the review was submitted. 

Datemodified: The date when the review was last modified.

# EXPLORATORY DATA ANALYSIS

From the basic analysis, the recipe dataset has a shape of 522,517 rows and 28 columns, while the review dataset has 1,401,768 rows and 8 columns. As illustrated in Figure 3.2, it is evident that the recipe data contains a significant number of missing values, particularly around the aggregated rating, review count, and recipe yield columns, which have data types of float, float, and object, respectively. In the review dataset, null values are present only in the review column, indicating that some users submitted only the rating score without providing review text.

The dataset appears to have been converted from a text format into a CSV file, suggesting the need for extensive preprocessing to ensure the content is in a usable format. The missing values and outliers will be carefully addressed later in this project, as improper handling could introduce noise and negatively impact subsequent analysis.
The collated data for the recipe dataset spans from August 1999 to December 2020, while the review data ranges from January 2000 to December 2020. This indicates that the platform started collecting user interactions four months later. A total of 3,673 recipes were published in 1999. In the year 2007, the number of recipes published was at its peak of over 70,000 which then reduces as the year increased, as of 2020, the number of recipes published were just about 1600.

Due to the limited computational efficiency, the dataset is stratified, data between 2013 and 2020 were selected.

![image](https://github.com/user-attachments/assets/770e4db4-d16d-4d10-bba5-dc10b354c1c9)

Figure: Number of recipes published in each year.

In the review dataset, most of the recipes were rated 5, suggesting that the users are very satisfied with the recipes.

![image](https://github.com/user-attachments/assets/ce6f6320-e68c-4d5b-9319-6f51daa46ff5)

Figure: A bar plot of rating distribution

# STATISTICAL SUMMARY FOR NUMERICAL AND CATEGORICAL FEATURES

From the statistical analysis of the numerical features, the aggregated rating has a mean of 4.63 and a low standard deviation of 0.64, which indicates that the recipes are generally rated well, and the ratings are clustered around the mean. This mostly suggests that users are either satisfied or biased towards giving higher ratings. The average review count is 5.23, with a median of 2, showing that most recipes have a very small number of reviews, while only a few garnered significant interests from users. The range of calories from 0 to 612,854.6 suggests the presence of extreme outliers. The high standard deviation of 1397.12 indicates significant variability in calorie content, which may result from various recipe categories with diverse calorie ranges. Other nutritional content, such as fat, saturated fat, cholesterol, sodium, carbohydrate, fiber, sugar, and protein, also show high standard deviations, indicating substantial variability in these nutritional features as well.

The recipe category that was mostly published out of 311 is dessert which has 62,072 recipes while lunch/snack and one dish meal has an entry of 32,586 and 31,345 respectively.

![image](https://github.com/user-attachments/assets/f520fee5-12f7-44c4-997f-10e6691292b1)

Figure: A plot of top 10 recipe category

# DATA PREPROCESSING

# DATA CLEANING

The data types of some features are incorrect and need to be converted to their appropriate data types. Features like ‘preptime’, ‘cooktime’, and ‘totaltime’ are currently of the object data type due to their format (e.g., PT24H for 24 hours, PT45M for 45 minutes, and PT2H20M for 2 hours and 20 minutes). To correct this, the hours were extracted, converted to minutes, and then added to the minutes if present, ensuring all times are recorded in minutes. Consequently, ‘preptime’, ‘cooktime’, and ‘totaltime’ were converted to integers (appendix A).

The review dataset contains some features which are not needed for the task, so these features were removed as their contribution is not required. Features such as ‘AggregatedRating', 'ReviewCount', 'RecipeYield', 'RecipeServings', 'Images' were dropped. In the review dataset, ‘DateSubmitted’ and DateModified’ were also dropped.
During analysis, some rows were found to be duplicated, likely due to multiple entries or system errors. These duplicates were removed. It was detected that some features such as RecipeIngredientPart, RecipeInstructions, and Keywords were combined into a single vector using the R programming language. This was evident because some features contained a ‘C’ function, which in R language represents “concatenation,” indicating that the data had been processed or stored in R. Consequently, these features needed to be cleaned and transformed into lists of strings (Appendix B).

# DEALING WITH MISSING VALUES AND OUTLIERS

Missing values in a dataset represent data that are either not recorded or null, often resulting from information loss during data collection. Outliers, on the other hand, are values that deviate significantly from the rest of the data, potentially skewing the distribution. In this project, missing values were removed rather than replaced. The replacement option was rejected because it is difficult to determine an appropriate substitute for the missing values. Outliers in numerical features (Figure 3.6) were removed using the interquartile range (IQR) method. Specifically, values below the lower limit and above the upper limit were removed. This method ensures that the dataset is cleansed of extreme values that could distort the analysis.
 
Mathematically,

IQR = Q3 -Q1	

Lower Limit = Q1 – 1.5 x IQR	

Higher Limit = Q3 + 1.5 x IQR	

# CONTENT-BASED FILTERING

# TEXT PREPROCESSING, TEXT CLEANING AND STOPWORD REMOVAL

To build a content-based filtering recommendation system, it is crucial that the text similarity is assessed on high-quality data. This involves cleaning, transforming, and preparing the text into a suitable format. The recipe dataset, which includes information about various recipes, was processed to ensure comprehensive coverage of relevant details. Specifically, three features— description, recipe ingredients, and keywords—were combined to form a new feature. This consolidated information provides a basic explanation of each recipe and requires thorough cleaning to avoid misinterpretation due to incorrect or inconsistent words.
 
The first step in preprocessing involved converting all text to lowercase. This normalization ensures that words like "Talk" and "talk" are treated as the same term, preventing unnecessary dimensionality increase. Punctuation marks and symbols were also removed, as they typically add noise to the data and are not needed for the analysis.
Contractions were expanded into their constituent words; for example, "won’t" was split into "will" and "not." Additionally, symbols that convey specific meanings in sentences were converted into their full-word equivalents. For instance, ">" was converted to "greater than," and "%" was converted to "percent."

Stop word removal is another critical component of text preprocessing. Common words such as "no," "of," "or," "had," and "has" do not contribute significant contextual meaning to the recipe descriptions and were therefore removed, this will also prevent high sparsity and dimensionality. The Natural Language Toolkit (NLTK) was employed to access its built-in list of stop words. During analysis, it was also observed that some descriptions ended with "Food.com," which did not contribute meaningful context. This term was added to the stop words list to eliminate any potential noise.

# 	WORD LEMMATIZATION
Lemmatization is a key process that reduces a word to its root meaning and at the same time give meaning to the word, it connects words to the like meaning or root word instead of root stem as in the case of stemming. The main goal is to keep and not lose the context of the sentence, this keeps the meaning of the word being described e.g. better: good. In this project the wordNetLemmatizer from the NLTK was employed (Appendix C).

# 	TEXT VECTORIZATION
The preprocessed text needs to be converted into a numerical format to facilitate further processing and enable machine understanding. The vectorization technique employed is TF-IDF, which transforms the text into numerical vectors. This approach focuses on the frequency of each word in the processed text and the number of processed texts in which a particular word appears, relative to the total number of processed texts. When a word appears in all the processed texts, its IDF value will be zero, resulting in a TF-IDF value of zero (Ma, 2016). Similarly, if a word does not appear in a particular processed text, the TF value will be zero, leading to a TF-IDF value of zero, as:
 
𝑇𝐹 − 𝐼𝐷𝐹 = (𝑇𝐹 × 𝐼𝐷𝐹)	(3.4)

The stratified data of recipes from 2013 to 2020, after being transformed using TF-IDF Vectorizer, results in a matrix with the shape (27,268, 49,555). This indicates that there are 27,268 rows corresponding to unique recipe IDs and 49,555 unique words. In cases where a word has a value of zero, it indicates that the word is not present in the processed description of that particular recipe ID.

For instance, it is shown that the term ‘chicken’ has a zero score in recipe 1, suggesting that the term does not appear in that recipe. However, it has a score of 0.238335 in recipe 10, indicating its presence in that recipe.

COSINE SIMILARITY

The similarity between vectors of each recipe is calculated using the cosine similarity which is mathematically the dot product of the vectors divided by their magnitude. Vectors with smaller angles produce a larger cosine value indicating a higher similarity.
 
 ![image](https://github.com/user-attachments/assets/f2a750ac-eb87-49da-b72d-188beee99103)

Cosine similarity for recipes

From the figure above, it can be observed that recipes 1 and 3 have a smaller angle between their vectors, which results in a higher similarity value. Consequently, the system will recommend recipe 3 to the user based on the higher similarity score when compared to recipes 2 and 4. Recipes 2 and 4 exhibit smaller similarity values due to the larger angles between their vectors. This similarity is calculated using the cosine similarity metric, as defined by equation 2.6. The cosine similarity is computed for each recipe relative to the other recipes, allowing the system to select the recipe with the highest similarity score to the requested recipe for recommendation to the user.

Upon examination, it is evident that the description, keywords, and recipe ingredients of these recommended items are similar to those of the requested recipe. This demonstrates how content- based recommendation systems work: by recommending items with content that closely matches the requested item (Messaoudi & Loukili, 2024).

COLLABORATIVE FILTERING
ITEM-ITEM COLLABORATIVE FILTERING

Collaborative filtering involves recommending recipes based on interactions between users and recipes, specifically utilizing ratings. To achieve this, a rating matrix is created. The dataset is stratified by selecting recipes with a rating count exceeding 100. The matrix is constructed by pivoting the DataFrame, where rows represent AuthorId and columns represent unique RecipeId.
 
Null values, where no rating has been provided, are replaced with zeros. The resulting matrix has a shape of (86789, 1065), indicating the dimensions for AuthorId and RecipeId.

For item-item collaborative filtering, Pearson correlation is employed to determine the pairwise correlations between recipes. The matrix demonstrates the strength and direction of the linear relationships that exist between the ratings of different recipes. A positive correlation signifies that the recipes are rated similarly by users, thereby suggesting these recipes to users. Conversely, a correlation of zero indicates no significant relationship in the ratings between the recipes.

Recommendations are based on scores calculated using a weighted average approach. The weights are only calculated for unrated items, as items that have already been rated by the user are generally not recommended back to them. The recommendation score for an item j, given that the user has rated item i, is calculated as follows:

 
Weighted average recommendation score = ∑𝑖∈rated(𝑟𝑢𝑖×corr𝑖𝑗)
∑𝑖∈rated corr𝑖𝑗
 
(3.5)
 

where:

rui is the rating given by the user u to the item i

corrij is the correlation between unrated item i by user and rated item j by the user.

The numerator calculates the weighted sum of the user's ratings, where the weights are the correlations between the unrated and rated items. The denominator normalizes the score. Higher scores indicate greater similarity between the unrated items and the rated items. This process facilitates the suggestion of items that are similar to the user’s preferences, thereby creating a more personalized recommendation experience.
 
 
![image](https://github.com/user-attachments/assets/0377a56b-d758-466c-920d-67532eb0b737)

Item-item collaborative filtering

# USER-ITEM COLLABORATIVE FILTERING

This approach involves suggesting recipes to users based on the relationships between similar users. Specifically, if User A rates recipes similarly to User B, the unrated recipes for User A will be suggested to User B. This is because both users have rated similar items in the same way . Similarity scores between users are computed using cosine similarity. The K-nearest neighbors’ algorithm is used to find users that have similar rating patterns.

![image](https://github.com/user-attachments/assets/fa50b5a1-1855-4d4b-8001-3fa9d11ee1e0)

User-recipe collaborative filtering
 
Given the highly sparse nature of the rating matrix, the Compressed Sparse Row (CSR) matrix format from Scipy is employed to store the matrices in a space-efficient manner, ensuring effective calculations. The algorithm used is a brute-force method which computes the distance between recipes based on cosine similarity of users as the metric. Recommendations are made based on the shortest distance between the recipe calculated by the algorithm. The algorithm selected the recipes with closest distance, based on the recipes rated by similar users. The Brute force algorithm searched for recipes similar to the query recipe vector. Basically, the recipe vectors which are ratings by users uses cosine similarity to measure the similarity with other recipe vector, the brute force search algorithm compares every of the recipe vector with the query recipe vector to select the closest recipe vector.

# DEEP LEARNING TECHNIQUE

# GRAPH NEURAL NETWORK

This technique is particularly noteworthy due to its capability to learn intricate information. In this study, the information from the recipe and review datasets is represented as a graph, where nodes contain embeddings and are connected by edges. A Graph Neural Network (GNN) is employed to predict user ratings for recipes, effectively tackle sparsity problem often encountered with matrix. This model predicts ratings for recipes for new users with high accuracy.

GNN captures all relationships between users and recipes, thereby generating personalized recommendations that align with user interests. By learning from the graph structure, GNN also incorporates information about new recipes and new users, surpassing the traditional methods used previously. This project extends beyond a homogeneous graph approach by utilizing a heterogeneous bipartite graph structure. This structure includes both user-recipe interactions and recipe-recipe interactions. In this setup, edges in the user-recipe interaction graph represent ratings, while recipes are linked based on their categories in the recipe-recipe interaction graph. This sophisticated approach enhances prediction accuracy and recommendation quality by leveraging the complex relationships within the data (Lavryk and Kryvenchuk, 2023).

In this study, a graph is considered with user nodes and recipe nodes, denoted as G = (U, R) where U represents the set of users and R represents the set of recipes. The graph includes a set of rating edges, ε, which connect the nodes. For a node u representing a user and a node r representing a
 
recipe, an edge (u, r) 𝜖 ε signifies the user-recipe interaction. This edge is directed and does not have a reverse connection. In addition to the graph structure, connections between recipes that belong to the same category are represented as undirected edges. For example, if ri, and rj are two recipes within the same category, their connection is represented as (ri, rj) ↔ (ri, rj). This undirected connection captures the relationship between recipes in the same category, facilitating a more nuanced analysis of recipe similarities and interactions.

![image](https://github.com/user-attachments/assets/0965bf01-3735-4b94-98ba-835e28f882fd)

Figure: User-Recipe graph illustration




# NODE EMBEDDING
This method effectively captures discrete information such as categories, descriptions, and keywords by representing them as continuous vectors. These discrete pieces of information are encoded into a lower-dimensional vector space that retains their essential details. In the heterogeneous graph, each node possesses its own attributes or feature information. Specifically, recipes include attributes like category and description, while users do not have any additional information due to the lack of user-specific data such as address or city. These attributes are represented as a matrix X𝜖ℝ|v| x m.
To create the recipe features vector, the recipe descriptions are encoded using the Sentence Transformer from Hugging Face. The 'all-MiniLM-L6-v2' model is employed for this encoding process (Reimers & Gurevych, 2019). The resulting embeddings are then concatenated with the
 
categorical features, which have been transformed using one-hot encoding. This combination generates the recipe features matrix.
User embeddings are initialized using an identity matrix which returns one in the diagonal and zero elsewhere due to the absence of user-specific information. This initialization approach reflects the lack of available data for user attributes.

![image](https://github.com/user-attachments/assets/1ec241a8-60bd-4c96-b060-a1804f376ed7)


Figure: Node Feature




# GRAPH STRUCTURE
The graph is composed of two types of nodes: recipe nodes and user nodes. Each recipe node includes embeddings representing feature information such as recipe category and description, while user nodes are initialized with an identity matrix. User nodes are connected to recipe nodes through rating edges, which reflect the ratings provided by users for recipes. To manage and track connections between nodes, each unique user ID and recipe ID is mapped to a new sequential integer ID. These mapped IDs are then linked by the rating edges, which connect user nodes with recipe nodes based on the ratings given. To create a complex graph, recipes are also linked based on their categories. Specifically, recipes with similar categories are connected with undirected edges (figure 3.14). Recipe categories are label-encoded, which assigns a numerical value to each unique category. Additionally, a tensor is created to index the edges between recipe nodes and user nodes, indicating which user nodes are connected to which recipe nodes and including the edge labels representing the ratings.
 
 ![image](https://github.com/user-attachments/assets/d1730f13-cd9c-45c7-ac5f-bcb97312cfc9)


Graph Structure for user-recipe

# GNN MODELS AND ARCHITECTURE

The study utilizes two models: GraphSAGE (Hamilton et al., 2017) and GAT (Veličković et al., 2018). These models were selected for their inductive learning capabilities and their methods of aggregating neighbor nodes using different techniques. GAT (Graph Attention Network) employs an attention mechanism to obtain node representations and computes target node embeddings through an aggregation and update function. In contrast, GraphSAGE (Graph Sample and Aggregation) samples relevant neighboring nodes and aggregates features using functions such as mean or LSTM.

The GraphSAGE model architecture in this study is designed by combining an encoder and a decoder. The encoder component is responsible for encoding node features into embeddings using GraphSAGE convolutional layers. This encoder is adapted to handle heterogeneous graphs using the ‘to_hetero’ function from PyTorch Geometric. The convolutional layers aggregate features from sampled neighboring nodes using the mean aggregation function. The architecture consists of three convolutional layers, each followed by ReLU activation functions applied sequentially. The edge decoder component decodes the embeddings produced by the encoder into edge scores or attributes, which are useful for predicting edge properties such as ratings and for recommendations. The fully connected layers in the decoder reduce the concatenated features of the connected nodes to generate the final predictions.

![image](https://github.com/user-attachments/assets/c911a8ab-bff5-4d6c-adf2-ab792f6c07f1)

Figure: the GraphSAGE model architecture including encoder and decoder.

The architecture of GAT is also designed in the same way, just that the number of fully connected layers in the decoder was increased by 1.

![image](https://github.com/user-attachments/assets/119aed90-00fc-4659-8936-bf3c59602f8c)


the GAT model architecture including encoder and decoder.

The encoder is designed to transform the node features through a series of graph convolution layer, the forward pass ensures that the node features are processed sequentially, and after each convolutional, the ReLu activation function applied inserts non-linearity into the model. The
 
decoder is responsible for predicting edge related features based on the encoded features. This architecture facilitates effective learning and prediction.

The model loss is evaluated using the Root Mean Square Error (RMSE), with a lower RMSE indicating better model performance. The training and validation sets are run through several epochs, and the epoch with the minimum RMSE is selected as the best model. This best model is then applied to the testing data. The predicted values are compared with the actual ratings to ensure that the predicted ratings are close to the true ratings.
3.8.5.	GNN MODEL EXPLANATIONS AND RECOMMENDATION
To further understand the node features that play a crucial role in predicting ratings, the explanation of the graph needs to be explored (Amara et al., 2022). The ‘CaptumExplainer’ algorithm was employed to compute these attributions, using ‘IntegratedGradient’ attribution method. This process aids in enhancing transparency and building trust in the model by elucidating the factors influencing the decisions made.
A ‘HeteroData’ object created by the explanation class contains masks for edges, node features, and attributes. The mask represents the attribution, with larger mask values indicating greater importance of the component to the prediction of the rating value. The employed explanation type is the model, which reveals the inner workings of the "black box" and explains how decisions are made, where the model predictions serve as the target.
The explanation method is also directed towards making recommendations. Neighbouring recipes that contribute more to the rating prediction are recommended based on the attribute values calculated by the explanation. The metric used for prediction is RMSE (Root Mean Square Error), and efforts are made to ensure that the RMSE is minimized through multiple epochs to achieve optimal prediction accuracy.
To test and recommend recipes to a user, recipes not yet rated by the user are randomly selected. Since the model is inductive, it can predict the rating values for these recipes. Subsequently, the attributions of all recipes contributing to the prediction are collected using the explanation. Recipes with higher attribution values are recommended to the user.
 
3.9.	HYBRID RECOMMENDATION SYSTEM

This method is important because it helps address the cold start problem and data sparsity (Çano and Morisio, 2017). It ensures that recommendations are based on a combination of content-based filtering and collaborative filtering (both user-item and item-item based collaborative filtering). The approach used in this project integrates the results from these different methods, ensuring that personalized recipes are recommended to all users including new users.

![image](https://github.com/user-attachments/assets/2890b195-d83c-4b34-8a9f-199453bf3eb1)


Hybrid Recommendation System

A more sophisticated approach employed in the study involved combining recommendations from the graph neural network (GNN) and content-based method. This approach ensures that more information is captured, further improving the accuracy of recipe recommendation to the user by considering both the interactions and similarities.

 ![image](https://github.com/user-attachments/assets/8d90b207-d709-47e7-8e02-c3eb981503cb)

Hybrid GNN-Content Based Recommendation System


# RESULTS AND DISCUSSIONS

# RESULT FROM THE CONTENT-BASED FILTERING.

After data preprocessing and cleaning, the data from 2013 to 2020 was vectorized using TF-IDF. The resulting matrix had dimensions of (27,268, 49,555). The matrices were compared using two techniques: the sigmoid kernel and cosine similarity. Recipe recommendations were based on these similarity scores, with higher scores indicating greater relevance between recipes.
The results of the comparison revealed that both techniques produced similar outcomes or recommendation for the same recipe. This approach effectively identified recipes with analogous processed information. Further analysis demonstrated that the recommended recipes shared similarities in their descriptions, keywords, and categories.

 ![image](https://github.com/user-attachments/assets/83bf7e2c-e524-4807-9213-9f173d47be13)

Recommendation using sigmoid kernel.
 
 ![image](https://github.com/user-attachments/assets/dc63dd79-33a5-40ca-8ddb-741a34d16e29)

Recommendation using cosine similarity.

# RESULTS FROM ITEM-ITEM COLLABORATIVE FILTERING

the correlation between recipes was measured using the Pearson correlation method. The weighted average score was used as the basis for recommending items. Recipes with higher scores were recommended to the user. Recommendations were made based on recipes that similar users had interacted with. The correlation between these recipes was measured, and recommendations were provided to the user accordingly.

 ![image](https://github.com/user-attachments/assets/c4dc94cf-1195-4154-a6f6-ecaa2cc020a4)


Item-Item Collaborative Filtering

# RESULTS FROM USER-ITEM COLLABORATIVE FILTERING

In this study, the K-Nearest Neighbors (KNN) algorithm was utilized to recommend items similar to a given query item. This algorithm relies on cosine similarity and a brute-force search technique. The distances and indices of the nearest neighbors to the query item determine the recommendations, with shorter distances indicating higher likelihoods of recommendation.

![image](https://github.com/user-attachments/assets/c33c892e-2267-4c54-9ae8-f9baec9111ea)

User-Item collaborative recommendation output

To further elucidate the spatial relationships between the query item and its neighbors, a dimensionality reduction technique using Principal Component Analysis (PCA) was applied. This method reduces the data's dimensionality to two dimensions while retaining as much variance as possible. The visual representation presented below illustrates the proximity of the nearest neighbors to the query point, based on the relative distances calculated by the KNN algorithm.
 
 
![image](https://github.com/user-attachments/assets/402e9631-9306-4322-a93e-2da73fe370ae)

Visualization of recommended recipe using PCA

# RESULTS FROM HYBRID RECOMMENDATION

The recommendations are displayed from content-based filtering and collaborative filtering approaches combined together. By leveraging the results from these systems, a hybrid recommendation system can mitigate the impact of sparse data in cases where there is a limited amount of data regarding user-item interactions. Additionally, cold start problem will be addressed by recommending items based on the user preferences that have been selected. The recommendations for using hybrid approach are seen below.

![image](https://github.com/user-attachments/assets/490eda56-dbff-43fe-baf1-a4f007c90645)

Results of hybrid recommendation system
 
# RESULT FROM GRAPH NEURAL NETWORKS (GNNS)

A heterogeneous bipartite graph was constructed, comprising recipe nodes and user nodes, which are interconnected through rating edges. To enhance the complexity of the graph, recipes within the same categories were also linked. The data utilized in this study encompassed recipes published between January 2014 and December 2020, along with user reviews for these recipes within the specified time frame, totaling 4,989 records. The dataset included 2,347 unique recipes and 954 unique users. Additionally, there were 163 distinct recipe categories, sauces are the most recipe categories with over 650 recipes, followed by desert and potato which are above 500 and 400 respectively.

![image](https://github.com/user-attachments/assets/881be5d8-7b8a-4d80-821d-63ede16f0707)

Top 20 recipe categories count

For recipe embedding, both descriptions and categories were employed. Recipe descriptions were processed using a sentence transformer, while recipe categories were encoded using label and one- hot encoding techniques. User embedding was represented by an identity matrix due to the lack of relevant information for a more detailed user representation. To facilitate the linkage of nodes, each unique recipe and user was assigned a unique ID. The recipe feature matrix was constructed by combining the unique recipe IDs with the unique categories, resulting in a tensor with dimensions of [2,347, 163].
 
The total linkage between the recipe and user nodes has a shape of “torch.Size([2, 4989])”, while the rating edges have a shape of “torch.Size([4989])”. The linkage that exists between the recipes has a shape of “torch.Size([2, 96787])”. During the graph construction, the links between recipes based on their categories were doubled. This doubling occurred because the links between recipes are undirected.
The resulting graph structure of the heterogeneous graph after construction is represented below.
HeteroData( User={ node_id=[954], x=[954, 954],
},
Recipe={ node_id=[2347], x=[2347, 547],
},
(User, RATING, Recipe)={ edge_index=[2, 4989], edge_label=[4989],
},
(Recipe, CATEGORY, Recipe)={ edge_index=[2, 193574] }, (Recipe, rev_RATING, User)={ edge_index=[2, 4989] }
)
 
 ![image](https://github.com/user-attachments/assets/56fc19cb-58ee-4222-ac98-7d26f578c4a3)


Heterogeneous Graph Strcuture of User-Recipe


The data was split into training, validation, and test sets using the RandomLinkSplit method, which ensures that there is no information leakage. The edges of the graph, specifically ('User', 'RATING', 'Recipe') and ('Recipe', 'CATEGORY', 'Recipe'), were divided in a ratio of 70:20:10, with 70% allocated to training, 20% to validation, and 10% to testing.
The hyperparameters used include Adam optimizer with a learning rate of 1×10-3. Training and validation were conducted over multiple epochs, and the best model was selected based on the epoch where no further decrease in the RMSE (Root Mean Square Error) value of validation was observed, the model with the lowest RMSE and no other problem of overfitting was selected. The objective was to minimize the RMSE value as much as possible.
For each GNN model utilized, the best-performing model was saved and subsequently applied to the test data. The performance of these models was then compared with the actual ratings to measure the RMSE value.
 
# MODEL EVALUATION

During training, the RMSE consistently decreased along with the validation RMSE. However, at a certain epoch, the validation loss ceased to decrease and began to increase. In this project, the best model was selected based on the lowest RMSE achieved by the validation loss, ensuring that the differences between the training RMSE and validation RMSE is at the minimum. Similarly, a correlated concept of graph-based recommendation system used on amazon data gave a lower RMSE which show the ability of Graph technique to make an accurate recommendation when compared with other methods and algorithm used on the dataset (Chang et al., 2024). In figure 5.9, it is observed that while the training loss continued to decrease, the validation loss stopped improving, indicating that further training would likely lead to overfitting. This suggests that the model would not generalize well to the unseen data as the training continues. GraphSAGE was trained over 1000 epoch, while the best result is at epoch 676. For GAT, it was train over 1500 epochs, the best RMSE value was at epoch 1113.

![image](https://github.com/user-attachments/assets/1d169cb5-1f3b-4c16-ad96-db32ab3b1b00)

Performance of GraphSAGE (RMSE)
 
 ![image](https://github.com/user-attachments/assets/d942a87a-b5d5-46cd-ac1f-adf858574e78)


Performance of GAT (RMSE)

While trying several possibilities, it was observed that the RMSE value decreased as the volume of data increased. Also, as the complexity of the graph increased, more information was learned, leading to more accurate predictions of ratings and lower RMSE. During this investigation, the use of a more complex and larger dataset resulted in reduction in RMSE, thereby improving the prediction accuracy. The result below was recorded when the data entries is 4989 with complex graph structure.

Model	Training
RMSE	Validation
RMSE	Test RMSE
GraphSAGE	1.4842	1.6840	1.6578
GAT	1.5913	1.7182	1.7183 

Table: Model Evaluation of complex heterogeneous graph

When the model was trained on a smaller dataset of approximately 2,275 entries, with a simple heterograph structure having only a linkage between the user and recipe based on rating, with no
 
interactions between recipes, the RMSE values were high, an higher RMSE results into a prediction of rating that is far from the actual ratings. The simple heterograph structure is shown below.
HeteroData( User={ node_id=[518], x=[518, 518],
},
Recipe={ node_id=[1234], x=[1234, 521],
},
(User, RATING, Recipe)={ edge_index=[2, 2275], edge_label=[2275],
},
(Recipe, rev_RATING, User)={ edge_index=[2, 2275] }
)


The result is represented below:

Model	Training
RMSE	Validation
RMSE	Test RMSE
GraphSAGE	1.5361	1.8693	1.8491
GAT	1.6192	1.9531	1.8261

Table Evaluation of simple heterogeneous graph

# RECOMMENDATIONS BY GNNS
The GraphSAGE model emerged as the most effective, exhibiting the lowest test RMSE compared to the GAT model. The lower RMSE on test data indicates superior accuracy because it performs well on the unseen. This recommendation technique facilitates the prediction of ratings for recipes that have not yet been rated by the user and utilizes explanations to select neighbouring recipes with higher attribution values as determined by the explanation method. This means that when a
 
user interacts or clicks on a recipe that has not been rated, the model can predict the rating and as well recommend recipes to the user.
To illustrate this, a recipe that has not been rated by the user, who had rated the most recipes, was chosen. In this instance, the recipe with a recipe ID of 517863 and a mapped ID of 889 was selected. The model predicted a rating of 3.98 for this recipe. Using the explanation method, attributes were ranked from highest to lowest, and the top 5 neighbouring recipes with the highest attribution values were recommended to the user. The results of the recommended items are shown below.
 
![image](https://github.com/user-attachments/assets/464e4911-b93e-442b-b455-213381365058)

Recommended recipe for recipe with id 517863

5.8.	HYBRID RECOMMENDATIONS (GNN + CONTENT BASED FILTERING)
The result from graph neural network model that uses message-passing between nodes to learn embeddings and content based which involves finding similarities are stacked together to make a more accurate personalized recommendation. In comparison with the study by Yijun et al., 2022, this recommendation puts new recipes into consideration by recommending new recipes that has not been rated by users.

![image](https://github.com/user-attachments/assets/6161c394-9125-4786-bde3-735b38046de0)

Figure: Recommended item using the developed hybrid for recipe id 517863.
 

CONCLUSION
In this study, various recommendation techniques have been explored, including content-based filtering, collaborative filtering, hybrid approaches, and advanced methods such as Graph Neural Networks (GNNs) for personalized user recommendations. Content-based filtering recommends recipes based on their descriptions, keywords, and categories. This method utilizes the TF-IDF (Term Frequency-Inverse Document Frequency) score to identify and suggest recipes with similar content, thus providing recommendations tailored to the specific characteristics of the recipes.

In contrast, collaborative filtering involves two main techniques: item-item collaborative filtering and user-item collaborative filtering. The item-item collaborative filtering method recommends recipes based on the correlations between items, using metrics such as Pearson correlation scores. Recipes with higher correlation scores are recommended to users. Additionally, user-item filtering, specifically K-Nearest Neighbors (KNN), finds the nearest neighbors in the recipe space. Recipes that are closer in the feature space are considered more similar and thus recommended. The use of Compressed Sparse Row (CSR) matrices optimizes the process by reducing memory usage and accelerating matrix operations, leading to more efficient and effective recommendation.

The Graph Neural Network (GNN) approach employs a complex method by constructing a graph that links user nodes to recipe nodes based on user ratings, with ratings represented as edges. And also recipes are connected to one another based on their categories, such that recipes within similar categories are linked together. This technique enables the creation of a complex graph structure, the model architecture including the encoder and decoder effectively predicts ratings for new recipes and as well recommends similar recipes to users.

The model architecture incorporates Graph Attention Networks (GAT) and GraphSAGE. The study found that larger data volumes correlate with lower Root Mean Square Error (RMSE) values, leading to more accurate predictions and recommendations. Among the architectures, GraphSAGE demonstrated superior performance on unseen data. To enhance understanding of the predictions, the CaptumExplainer was employed to reveal neighboring recipes that contributed to the predicted ratings. These neighboring recipes are then recommended to users, providing insight into the rationale behind the predictions.

Finally, a more robust hybrid recommendation system was developed which combine the outputs of content-based filtering and Graph Neural Network to produce a better personalized recommendations compared to the commonly used hybrid recommendation system that combine content based and collaborative filtering. This hybrid approach addresses the data sparsity issue inherent in collaborative filtering and the cold start problem.


REFERENCES
Abdalla, H.I., Amer, A.A., Amer, Y.A., Nguyen, L. and Al-Maqaleh, B. (2023) ‘Boosting the item- based collaborative filtering model with novel similarity measures’, International Journal of Computational Intelligence Systems, 16(1), p. 123.

Aggarwal, C.C. (2016) 'Model-based collaborative filtering', in Recommender Systems: The Textbook, pp. 71–138.

Alotaibi, S., Altwoijry, N., Alqahtani, A., Aldheem, L., Alqhatani, M., Alsuraiby, N., Alsaif, S. and Albarrak, S. (2022) ‘Cosine similarity-based algorithm for social networking recommendation’, International Journal of Electrical and Computer Engineering (IJECE), 12, p. 1881. doi: 10.11591/ijece.v12i2.pp1881-1892.

Amara, K., Ying, R., Zhang, Z., Han, Z., Shan, Y., Brandes, U., Schemm, S. and Zhang, C. (2022) ‘Graphframex: Towards systematic evaluation of explainability methods for graph neural networks’, arXiv preprint arXiv:2206.09677.

Budhi, G., Chiong, R., Pranata, I. and Hu, Z. (2021) 'Using machine learning to predict the sentiment of online reviews: a new framework for comparative analysis', Archives of Computational Methods in Engineering, 28, pp. 10. doi: 10.1007/s11831-020-09464-8.

Burke, R. (2002) ‘Hybrid recommender systems: survey and experiments’, User Modeling and User-Adapted Interaction, 12, pp. 331–370.

Çano, E. and Morisio, M. (2017) ‘Hybrid recommender systems: A systematic literature review’,
Intelligent Data Analysis, 21(6), pp. 1487–1524.

Chang, et al., 2024. 'Building Amazon Recommendation Systems with Graph Neural Networks'. Medium. Available at: https://medium.com/p/building-amazon-recommendation-systems-with- graph-neural-networks [Accessed 8 September 2024].
 
Do, H.Q., Le, T.H. and Yoon, B. (2020) ‘Dynamic weighted hybrid recommender systems’, 2020 22nd International Conference on Advanced Communication Technology (ICACT), pp. 644-650. IEEE.

Do, M.P.T., Nguyen, D.V. and Nguyen, L. (2010) 'Model-based approach for collaborative filtering', in Proceedings of the 6th International Conference on Information Technology for Education, pp. 217–228.

Gao, C., Zheng, Y., Li, N., Li, Y., Qin, Y., Piao, J., Quan, Y., Chang, J., Jin, D., He, X. and Li, Y. (2021) ‘Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions’.

Glauber, R., Loula, A.C. and Rocha-Junior, J.B. (2013) ‘A mixed hybrid recommender system for given names’.

Gong, S. (2010) ‘A Collaborative Filtering Recommendation Algorithm Based On User Clustering And Item Clustering’, Journal of Software, 5(7), pp. 745-752. doi: 10.4304/jsw.5.7.745-752.

Guo, A. and Yang, T. (2016) ‘Research and improvement of feature words weight based on TF- IDF algorithm’, in 2016 IEEE Information Technology, Networking, and Electronic and Automation Control Conference, pp. 415–419, Chongqing, China.

Hamilton, W.L., Ying, R. and Leskovec, J., 2017. 'Inductive representation learning on large graphs', in Proceedings of the 31st International Conference on Neural Information Processing Systems, pp. 1025–1035.

Hammond, D.K., Vandergheynst, P. and Gribonval, R. (2011) ‘Wavelets on graphs via spectral graph theory’, Applied and Computational Harmonic Analysis, 30(2), pp. 129-150.

Javed, U., Shaukat, K., Hameed, I.A., Iqbal, F., Alam, T.M. and Luo, S. (2021) 'A review of content-based and context-based recommendation systems', International Journal of Emerging Technologies in Learning (iJET), 16(3), pp. 274–306.
 
Jiao, S., Zhang, Y. and Hara, T. (2024) 'Multi-task multi-modal graph neural network for recommender system', Research Square. doi: 10.21203/rs.3.rs-4382803/v1.
Jones, Q., Ravid, G. and Rafaeli, S. (2004) 'Information overload and the message dynamics of online interaction spaces: A theoretical model and empirical exploration', Information Systems Research, 15(2), pp. 194–211. doi: 10.1287/isre.1040.0023.

Kang, G., Tang, M., Liu, J., Liu, X. and Cao, B. (2016) ‘Diversifying web service recommendation results via exploring service usage history’, IEEE Transactions on Services Computing, 9.

Kumar, M., Yadav, D., Singh, A. and Kr, V. (2015) 'A movie recommender system: MOVREC',
International Journal of Computer Applications, 124, pp. 7-11. doi: 10.5120/ijca2015904111.

Lavryk, Y. and Kryvenchuk, Y. (2023) ‘Product recommendation system using graph neural network’, in Emmerich, M., Vysotska, V. and Lytvynenko, V. (eds.) Proceedings of the Modern Machine Learning Technologies and Data Science Workshop (MoMLeT&DS 2023). Lviv, Ukraine, 3 June 2023. CEUR Workshop Proceedings, 3426, pp. 182-192. CEUR-WS.org.

Ma, K. (2016) ‘Content-based Recommender System for Movie Website’.

Mai, J., Fan, Y. and Shen, Y. (2009) 'A neural networks-based clustering collaborative filtering algorithm in e-commerce recommendation system', in Proceedings of the 2009 International Conference on Web Information Systems and Mining, pp. 616–619, June 2009.

Manjula, R. and Chilambuchelvan, A. (2016) 'Content-based filtering techniques in recommendation systems using user preferences', International Journal of Innovations in Engineering and Technology (IJIET), 7(4), pp. 149.

Mazlan, I., Abdullah, N. and Ahmad, N. (2023) ‘Exploring the Impact of Hybrid Recommender Systems on Personalized Mental Health Recommendations’, International Journal of Advanced Computer Science and Applications, 14(6).
 
Messaoudi, F. and Loukili, M. (2024) 'E-commerce personalized recommendations: a deep neural collaborative filtering approach', Operations Research Forum, 5. doi: 10.1007/s43069-023-00286- 5.

Mittal, N., Nayak, R., Govil, M.C. et al. (2010) 'Recommender system framework using clustering and collaborative filtering', in Proceedings of the 3rd International Conference on Emerging Trends in Engineering and Technology, pp. 555–558, November 2010.

Panyatip, T., Kaenampornpan, M. and Chomphuwiset, P. (2023) 'Conceptual framework of recommendation system with hybrid method', Indonesian Journal of Electrical Engineering and Computer Science, 31(3), pp. 1696-1704. doi: 10.11591/ijeecs.v31.i3.pp1696-1704.

Pham, M.C., Cao, Y., Klamma, R. et al. (2011) 'A clustering approach for collaborative filtering recommendation using social network analysis', Journal of Universal Computer Science, 17(4),
pp. 583–604.

Pradeep, N., Mangalore, K.K.R., Rajpal, B., Prasad, N. and Shastri, R. (2020) 'Content-based movie recommendation system', International Journal of Research in Industrial Engineering, 9(4), pp. 337–348. doi: 10.22105/riej.2020.259302.1156.

Rebelo, M.Â., Coelho, D., Pereira, I. and Fernandes, F. (2022) ‘A New Cascade-Hybrid Recommender System Approach for the Retail Market’, in Innovations in Bio-Inspired Computing and Applications, Lecture Notes in Networks and Systems, 419. Springer, Cham. doi: 10.1007/978- 3-030-96299-9_36.

Reimers, N. and Gurevych, I., 2019. 'Sentence-BERT: Sentence embeddings using siamese BERT- networks', CoRR, abs/1908.10084. Available at: http://arxiv.org/abs/1908.10084.

Roy, D. and Dutta, M. (2022) 'A systematic review and research perspective on recommender systems', Journal of Big Data, 9.
Sajannavar, S., Dharwad, J. and Tandale, P. (2022) 'Online shopping - an overview', International Journal of Computer Engineering and Applications, 15, pp. 51–56. doi: 10.30696/IJCEA.XV.XII.2021.79-84
 
Sarwar, B., Karypis, G., Konstan, J. and Riedl, J. (2001) ‘Item-based Collaborative Filtering Recommendation Algorithms’, Proceedings of the 10th International World Wide Web Conference, pp. 285-295.

Shengq, W., Huaizhen, K., Chao, L., Wanli, H., Lianyong, Q. and Hao, W. (2020) ‘Service Recommendation with High Accuracy and Diversity’, Wireless Communications and Mobile Computing.

Shuman, D.I., Narang, S.K., Frossard, P., Ortega, A. and Vandergheynst, P. (2013) ‘The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains’, IEEE Signal Processing Magazine, 30(3), pp. 83-98.

Singla, R., Gupta, S., Gupta, A. and Vishwakarma, D.K. (2020) 'FLEX: a content-based movie recommender', in 2020 International Conference for Emerging Technology, INCET 2020. doi: 10.1109/INCET49848.2020.9154163.

Tata, S. and Patel, J.M. (2007) ‘Estimating the selectivity of tf-idf based cosine similarity predicates’, ACM SIGMOD Record, 36(2), pp. 7–12. doi: 10.1145/1328854.1328855.

Tian, Y., Zhang, C., Guo, Z., Huang, C., Metoyer, R. and Chawla, N., 2022. 'RecipeRec: A heterogeneous graph learning model for recipe recommendation', in Proceedings of the 31st International Joint Conference on Artificial Intelligence (IJCAI), pp. 3441–3447. doi: 10.24963/ijcai.2022/478.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y., 2018. 'Graph attention networks', in International Conference on Learning Representations, pp. 1–12.
Vivek, M., Manju, N. and Vijay, M. (2023) 'Machine learning based food recipe recommendation system', Proceedings of the International Conference on Emerging Technologies in Computing

Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C. and Yu, P.S., 2020. 'A comprehensive survey on graph neural networks', IEEE Transactions on Neural Networks and Learning Systems, 32(1), pp. 4–24. doi: 10.1109/TNNLS.2020.2978386.
 
Yi, B., Shen, X., Zhang, Z., Zhang, W. and Xiong, N. (2005) 'Deep matrix factorization with implicit embedding for recommendation systems', IEEE Transactions on Industrial Informatics,
pp. 1-1. doi: 10.1109/TII.2010.2893284.

Zhang, J. (2023) Innovative food recommendation systems: a machine learning approach. PhD thesis, Brunel University London. Available at: http://bura.brunel.ac.uk/handle/2438/26643 [Accessed 23 June 2024]

