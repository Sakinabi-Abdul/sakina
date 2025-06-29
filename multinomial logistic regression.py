from sklearn.model_selection import train_test_split
from sklearn import datasets , linear_model ,metrics

digits =datasets.load_digits()
X = digits.data
Y = digits.target

X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=0.40, random_state=1)
reg =linear_model.LogisticRegression(max_iter=10000, random_state=0)
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)

# Print the accuracy of the model
print(f"Accuracy of the multi-class logistic regression model is {metrics.accuracy_score(Y_test, Y_pred) * 100:.2f}%")