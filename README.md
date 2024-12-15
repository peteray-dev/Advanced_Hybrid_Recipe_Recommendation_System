# Recipe_Recommendation_App

# A HYBRID RECOMMENDATION SYSTEM: ENHANCING OPTIMAL ALGORITHMIC
# PERFORMANCE (Food Recipes Recommendation System)

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
Recipeingredientparts: The names of the ingredients required to make the recipe. Aggregatedrating: The average rating of the recipe (i.e., the sum of the ratings divided by the total number of times the recipe was rated).
Reviewcount: The total number of reviews for the recipe.
Calories: The caloric content of the recipe.
Fatcontent: The fat content in the recipe. Saturatedfatcontent: The saturated fat content in the recipe. Cholesterolcontent: The cholesterol content in the recipe.
Sodiumcontent: The sodium content in the recipe. Carbohydratecontent: The carbohydrate content in the recipe. Fibercontent: The fiber content in the recipe.
Sugarcontent: The sugar content in the recipe. Proteincontent: The protein content in the recipe. Recipeservings: The number of servings the recipe yields.
 
Recipeyield: The yield of the recipe (e.g., number of pieces). Recipeinstructions: Step-by-step instructions for preparing the recipe. Review Dataset:
This dataset consists primarily of information about user interactions with the recipes. The features include:
Reviewid: The unique identifier for each review made by a user.
Recipeid: The unique identifier for the associated recipe. Authorid: The unique identifier for the user who made the review. Authorname: The name of the user who submitted the review.
Rating: The rating the user gave to a recipe, ranging from 1 to 5.
Review: The user's comments about the recipe. Datesubmitted: The date when the review was submitted. Datemodified: The date when the review was last modified.

