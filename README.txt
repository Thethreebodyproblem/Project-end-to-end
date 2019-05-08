In this zip file, files ended with .pkl are the objects required for the scipt in order to make prediction.


-----
Slalom.ipynb:
This is the offical notebook with the detail explaination of all the methology. If you want to know more about the detail of
this project, you can kind them in this file
-----
report.ppt:
This is the PPT which can give you an overview about this project
-----
model_online_deploy.py:
this is the script which can run in the command line to make prediction.
It has a parameter called PREDICTION and this can control the type of prediction (probability or not)
-----
IpAddress_to_Country.csv:
This is the lookup table we use to convert IP into country for the new data
-----
test.csv:
This is the data we want to predict. In the real-world setting, it can be replaced by the new data. But please
keep the columns name and file name the same
-----
browser LB.pkl, Country LB.pkl, sex LB.pkl, source LB.pkl
Those are the LabelEncoder for the category variable in the data. It is pre-trained encoder to convert categorical variable
into discrete and machine learning model readable format
-----
model.pkl:
This is the random forest classifier. 


########################################
About the problems for this project:

1.For each user, determine their country based on the numeric IP address.
I try the brute force approach at first. For each IP, I iterate the IpAddress_to_Country csv to find the
matched one. But its time complexity is so high.
So I implement an algorithm which borrow the concept of dynamic programming to do so. And the most 
optimized time complexity is O(N)

2.What evaluation metrics did you use to select a model and why?
I used roc_auc_score to evaluate the model
Since this is unbalanced problem, accuracy score is not a good fit. Beside from roc_auc_score, I could also
use F-1 and Kappa score.
But for the model training part, I use SMOTE to deal with this problem.

3.How would you use this model to detect fraud in real-time?
This python script enable you to predict batch data. 
Run the Slalom.py in the command line. Make sure the you have the objects in the zip file stored in right
file path. 
The final prediction is in the file called output.csv
