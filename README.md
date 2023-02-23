# Using Promos to Increase Average Spend

![pexels-engin-akyurt-1435750](https://user-images.githubusercontent.com/78283026/220493020-1547af51-c134-44d7-bc01-8896f8a3354e.jpg)

### Project Definition
Anyone who knows me would tell you that I can always be found with a Starbucks coffee. My drink, a grande latte with two raw sugars, is an essential part of my morning, afternoon, sometime evening, routine. Therefore, I jumped at the chance to dive into some Starbucks data. 

The data provided by Starbucks consists of three json files: portfolio, profile, transcript. The portfolio dataset consist of the promos that are offered to customers. The profile dataset contains demographic information for each of the reward members. Finally, the transcript dataset consists of every instance that a reward member received, viewed, and completed a promo offer, as well as each transaction made by a reward member. 

Using the data provided by Starbucks, I attempted to build a model to predict if a promo would successfully sway a reward member to spend more  than their average transaction amount to complete a promo. I will be using the accuracy metric to determine if my model is successful.

### Exploratory Data Analysis

The portfolio dataset consists of ten records. Each record corresponds with a promo that a reward member can receive. As can be seen in Figure 1 below, the portfolio dataset contains information regarding the channels used to inform the reward member of the promo, how much needed to be spent to complete the promo, how much each reward was worth, and more.

<p align="center">
<img width="864" alt="figure 1" src="https://user-images.githubusercontent.com/78283026/220493836-37a9a4dd-f548-4e5b-bc78-126a3baec55f.png"><br />
figure 1 - portfolio dataset
</p>

The profile dataset contains demographic information for each of the rewards members. Upon exploring this dataset, we can make a couple of observations about this group of rewards members. For example, we can see that a majority of reward members are between the ages of 40 and 80 (see figure 2). However, there appears to be a large group of reward members that are between 100 and 120. Since it is improbable that such a large portion of the rewards members are almost 120, it is safe to assume that the age metric is not reliable.

<p align="center">
<img width="421" alt="figure 2" src="https://user-images.githubusercontent.com/78283026/220494169-1e2a0537-8b59-4530-bcba-8dbda948a4af.png"><br />
figure 2 - Age of Customers Histogram
</p>

Another observation about the reward members is that most became a member in 2018 (see figure 3).

<p align="center">
<img width="411" alt="figure 3" src="https://user-images.githubusercontent.com/78283026/220494293-e4ea6b39-8cfe-4262-abd6-2f5ffff41372.png"><br />
figure 3 - Date of Membership Histogram
</p>

More reward members are men vs. women (see figure 4).

<p align="center">
<img width="418" alt="figure 4" src="https://user-images.githubusercontent.com/78283026/220494432-f493deaa-b1a2-496b-9f4d-af02f989ac46.png"><br />
</p>

Finally, most reward members earn less than $80,000 per year (see figure 5).

<p align="center">
<img width="418" alt="figure 5" src="https://user-images.githubusercontent.com/78283026/220494505-574be7f8-31d4-4487-a51a-7271ad88ed2b.png"><br />
figure 5 - Income Histogram
</p>

The last data set to explore is the transaction dataset. This dataset is an event log and tracks transactions, offers received, offers viewed, and offers completed. A majority of events in this log are transactions (see figure 6). 

<p align="center">
<img width="428" alt="figure 6" src="https://user-images.githubusercontent.com/78283026/220494740-aea67277-9e0d-4735-a2b9-3647998f9c82.png"><br />
figure 6 - Event Distribution
</p>

The next step in our data analysis will be to preprocess the data so that it is usable for our analysis.

### Methodology

Before I could build my model, I needed to create a dataset where each record showed the reward members demographic information, the promo information, and the associated transcript information. 

I started my data preprocessing with the portfolio dataset. Within the portfolio dataset, there is a column entitled channels. Each row within this column contains a list of the channels that each promo is advertised on. These channels include web, email, mobile and social. Since lists cannot be interpreted by models, I created a column for each of the channels and used a binary indicator to indicate if the promo was promoted on that channel (see figure 7). 

<p align="center">
<img width="428" alt="figure 7" src="https://user-images.githubusercontent.com/78283026/220495939-2b6e1f14-bc6b-4ebb-a547-56f58f670b19.png"><br />
figure 7 - channels transformation
</p>

My next step was to transform the offter_type (bogo = 0, informational = 1, or discount = 2) from a string variable to an integer variable (see figure 8).

<p align="center">
<img width="684" alt="figure 8" src="https://user-images.githubusercontent.com/78283026/220496244-e4887e3c-7dec-4e02-a1a5-c2779dbf2500.png"><br />
figure 8 - ofter_type transformation
</p>

Once I had completed the preprocessing of the portfolio dataset, I started transforming the transcript dataset. I separated the transcript dataset into four different datasets based on event type (transaction, offer received, offer viewed, and offer completed). Each of the new datasets contained a value column. This column contained a dictionary of data points associated with the event. For transaction event types, it contained the amount spent on the transaction. The datapoint is used to create a new dataframe containing each reward member's average transaction amount. For the offer received and the offer viewed events, the value column contained the offer_id of the promo which would be used to identify which offer the reward member received from the portfolio dataset. Finally, the value column for the offer completed events contain the offer_id, a datapoint which can be used to connect it to the portfolio dataset, and the reward amount for completing the promotion. Once the dictionary datapoint for each event type has been broken out into individual columns. The event type data frames are merged into a dataframe called promos resulting in the dataframe below (see figure 9).

<p align="center">
<img width="824" alt="figure 9" src="https://user-images.githubusercontent.com/78283026/220497009-1a82572c-9930-4ef8-adb0-1684258fced7.png"><br />
figure 9 - promo dataframe
</p>

The next step is to merge the profile dataset, the portfolio dataset, and the newly created avg transactions dataset to the new promo dataset. Before I do this, however, I dropped the age datapoint from the profile dataframe since it is unreliable as discussed above. I also dropped any profiles where gender is null (this consequently also removed all null values from the income column) since nulls were a relatively small portion of members. 

The last step, before the model can be created, is to create a column that indicates if the promo was successful called promo_worked. For the purpose of this analysis, a promo is considered to have “worked” if an offer is viewed, completed, and the offer had a difficulty (how much needed to be spent to receive the reward) that was greater than the members average transaction amount.

I divided my data into two datasets. A training set and a testing set. This allowed me to train my models on 80% of the data and reserve 20% of the test to test the models predictions. 

For the model, I decided to run a basic decision tree classifier from the scikit-learn. I used utilzied cross validation and tuning parameters to determine the optimal max depth of the decision tree. After fitting my model, I used a test subset of my data to gauge the models performance. The model performed with an accuracy score of 1 (i.e. 100%) when its predictions were compared to the test data.

### Results
The basic decision tree model perfomred with an accuracy score of 1. However, accuracy is not the best metric to evaluate this model. This is because the test dataset contain more instance where the promo did not work versus did work (18224 vs. 2315). Due to this disparity, F1 score is a better metric than accurary since F1 score uses precision and recall to accomodate for unbalanced options. The F1 score for this model was also 1 (see figure 13 below). This means that we are able to accurately predict when a promo would get a reward member to spend more than their average transaction. 

<p align="center">
<img width="384" alt="figure 13" src="https://user-images.githubusercontent.com/78283026/220931802-8f8ea9ad-86a2-4d42-9a5d-2ffc708085d5.png"><br />
figure 13 - model comparison
</p>

The decision tree model I was still able to glean some insight by taking a deeper dive into the records where it was determined that the promo was successful in getting the member to spend more than they do on average.

For example, the average transaction amount for reward members was $16.02, and the dataset contains numerous outliers above the third quartile range with the max avg transaction over $400 (see figure 10).

<p align="center">
<img width="487" alt="figure 10" src="https://user-images.githubusercontent.com/78283026/220497812-cd3ad420-ec93-4df6-8e1f-780af6cf0619.png"><br />
figure 10 - avg transactions boxplot & mean
</p>

When the dataset is filtered for only records where the promo was successful, however, the average transaction amount drops to $6.18 with the max average transaction being just $19.98 (see figure 11).

<p align="center">
<img width="502" alt="figure 11" src="https://user-images.githubusercontent.com/78283026/220498481-e1d33ff1-476d-4340-a52d-e598123dfbcb.png"><br />
figure 11 - promo-worked mean and avg transaction histogram
</p>

The difference in the mean of the average transactions indicates that a promo is more effective on consumers who on average spend less. This is further supported when looking at the distribution of difficulty (how much needed to be spent in order to complete the offer) with most being willing to spend $10 or less but only a small portion spending $20 to claim a reward (see figure 12).

<p align="center">
<img width="600" alt="figure 12" src="https://user-images.githubusercontent.com/78283026/220498618-a82f6a81-b3fe-4e9c-9003-2d4aecfc54f5.png"><br />
figure 12 - difficulty distribution
</p>

When taken together, it becomes clear that members who on average spent less per transaction were more likely to be swayed into spending more by a promo. This suggests that the purchasing patterns of those who on average spend more per transaction are more habitual and less likely to be swayed by promotions.

### Conclusion

In the end, we were able to pull out characteristics from these dataset to determine if a promotional offer is likely to sway a reward member into spending more than they typically do on average. Promotional offers are most effective on members who average spending less than $10 per transaction.

Further enhancements to this analysis can be made in further research efforts. For example, further preprocessing of the data prior to building the model may be required to compensate for the overfitting seen in this experiment. Also, other models can be introduced to see if they perform better than the decision tree model used in this experiment.

### References

"Brown Starbucks Paper on Gray Wooden Surface". Engin Akyurt. https://www.pexels.com/photo/brown-starbucks-paper-on-gray-wooden-surface-1435750/

N B, Harikrishnan. "Confusion Matrix, Accuracy, Precision, Recall, F1 Score". https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd

"Starbucks Capstone Challenge". Starbucks & Udacity. https://www.udacity.com/
