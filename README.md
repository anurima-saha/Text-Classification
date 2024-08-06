# Text-Classification
In this project, wine reviews have been used to determine the type of wine training on imbalanced an dataset using classification algorithms like SVM, Naive Bayes and Random Forest Classifier. Neural Network (CNN, RNN and LSTM) and LLM models (DistilBERT and RoBERTa) were also used followed by error analysis using SHAP.

## Overview:
We have been provided with a wine reviews dataset with two columns: “review_text” and “wine_variant” and the goal is to create a wine recommendation system using test classification.
#### Data:
* Target variable – ‘wine_variant’
* Categories – 8 Types - 'Pinot Noir', 'Sauvignon Blanc', 'Cabernet Sauvignon', 'Chardonnay', 'Syrah', 'Riesling', 'Merlot', 'Zinfandel'
* Train data – 10000 observations were split into test set of sample size 25% (2500). Stratified sampling used for appropriate representation of above-mentioned classes. An additional 
  validation data with 5000 observations has been used.
* Distribution – In percentage
  >![image](https://github.com/user-attachments/assets/772877d8-cd17-4014-bb28-6bb1cc005dc6)
  >![image](https://github.com/user-attachments/assets/a328d8b2-2e42-419c-ac37-dd28bdcc8df2)
## Models and Algorithms
#### Embedding:
* TF-IDF vectorization
* Latent Semantic Analysis
* Sentence Transformer (all-mpnet-base-v2)
* torchtext.vocab
  <br>
#### Alogorithms:
##### Supverised ML
* Linear and Non-linear SVM
* SDG Classifier
* Multinomial Naive Bayes
* Random Forest Classifier
##### Neural Network
* CNN
* LSTM
##### LLM
* DistilBERT
* RoBERTa
## Conclusion
From the above results we have the four best classifier along list in the order of descending macro average f1 score on validation set:
1. RoBERTa (0.80)
2. DistilBERT (0.79)
3. TFIDF Vectorization + Linear SVC (with hyperparameter tuning) (0.78)
4. CNN (0.77)
We can conclude two things from the above analysis:
1. Given the size of the training set, the transfer learning algorithms(RoBERTa and DistilBERT) are likely to provide much better results as seen in the table above.
2. Given the class imbalance in the dataset, the best way to group the categories is on the basis of domain knowledge as stated above. Grouping on the basis of taste and flavour is more appropriate when building a wine recommendation system rather looking at the distribution of target variables. This has led to a significant improvement in results improving classification accuracy from low 70s to almost 80%.
3. Although our model has shown a significant improvement in results from the baseline SVC model, the macro f1 score does not go above 80% even after working with
multiple models. This is a clear indication that we need more training data to improve our classification report.

## Error Analysis
We have used the RoBERTa model for performing error analysis using SHAP. We have taken a sample of 30 mis-predicted observations from the provided test set of sample size 500 for this analysis. We will look into a few samples for our report, for a model detailed analysis please refer to the code.

#### Example 1: “Medium to Full-bodied Reds” classified as “Bold Reds”
While words like “light” and “oak” incline the results towards “Medium to Full-bodied Reds”, the final outcome seems to influenced by the use of “powerful”, “refrain” and “berries”.
>![image](https://github.com/user-attachments/assets/b8ba2752-4fa3-44c0-81a7-29b55016251f)

#### Example 2: “Medium to Full-bodied Reds” classified as “Bold Reds”
In this example we see that the use of words like, “TONS” and “more fruit” has pushed the classifier to predict “Bold Red”
>![image](https://github.com/user-attachments/assets/8138acb9-f1f9-4dc7-84cf-d7222de99ad6)

#### Example 3: “Bold Reds” classified as “Medium to Full-bodied Reds”
In the given scenario, the word “medium” clearly influences the result
>![image](https://github.com/user-attachments/assets/97ab1b25-db08-4a8e-893a-216e29253f2e)

#### “Light-bodied, Crisp Whites” classified as “Full-bodied Whites”
The use of the word “champagne” which is a “Full-bodied white” has stirred the prediction to be as such.
From the above analysis we see errors that are primarily domain knowledge related. However, in the reviews we also have text that are redundant and do not contribute to the classification with respect to taste of quality of wine as seen below. Hence, a recommendation from this would to carefully curate samples that are used to train the wine-recommendation model in order to obtain more accurate results.
>![image](https://github.com/user-attachments/assets/b183f4f0-de95-4447-9d9d-a8186e30a5f9)

For more details please refer to [Project Report](https://github.com/anurima-saha/Text-Classification/blob/main/Project%20Report.pdf)


  

