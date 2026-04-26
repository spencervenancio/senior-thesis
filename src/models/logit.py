def logistic_regression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=300, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    return clf