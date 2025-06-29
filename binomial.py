from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load the dataset
X,Y =load_breast_cancer(return_X_y=True)
# Split the dataset into training and testing sets
X_train ,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=23)
# Create a logistic regression model
clf =LogisticRegression(max_iter=10000,random_state=0)
clf.fit(X_train,Y_train)
# calaculate the accuracy of the model
acc= accuracy_score(Y_test,clf.predict(X_test))*100
print(f"Accuracy of the breast cancer dataset is {acc}")