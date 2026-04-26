def logistic_regression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from src.utils import load_mnist
    X_train, _, y_train, _ = load_mnist()
    clf = LogisticRegression(max_iter=300, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    return clf