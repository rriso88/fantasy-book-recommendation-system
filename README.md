# fantasy-book-recommender

The objective of this project is to create a fantasrecommendationy book recommendation system using a hybrid of content and collaborative-based filtering.  I used user ratings and descriptions of over 40,000 fanatasy books from the goodreads.com API as the data source for this project.  
  
To achieve content-based filtering, I conducted natural language processing of book descriptions using spaCy. Next, I performed topic modelling with Gensim using Latent Dirichlet Allocation (LDA) to reduce the description text to five main sub-genres within fantasy: young adult, epic, supernatural, science fiction, and urban fantasy.
Content-based recommendations are given by taking cosine similarities between vectors of topic weights from the LDA reduction.

To generate collaborative-based recommendations based off of previous books a reader has read and enjoyed, I used a matrix of book ratings across 5,000 Goodreads users.  
I employed the Suprise library in Python to estimate user ratings. In the test set, the Singular Value decomposition (SVD) algorith had the lowest MAE of .656, meaning it was able to predict how many starts a user would rate a book within .656 points of the actual rating.  

Finally, I designed an interactive Flask app wherein a user can enter their ID and receive book recommendations tailored to their own likes and dislikes.

Future work includes acquiring additional user ratings to improve the performance of the collaborative-based filtering model, employing a more sophisticated algorithm and additional NLP for content-based recommendations, and posting an improved Flask app to Heroku for data visualization and real-time recommendations.
