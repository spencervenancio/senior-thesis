def make_loco_worker(model_factory, X_train, X_test, y_train, y_test, full_acc):
    """Returns a LOCO worker function closed over the given data and model factory."""
    def loco_one_feature(j):
        X_train_j = X_train.copy()
        X_test_j  = X_test.copy()
        X_train_j[:, j] = 0
        X_test_j[:, j]  = 0

        clf_j = model_factory()          # fresh instance every call
        clf_j.fit(X_train_j, y_train)
        j_acc = accuracy_score(y_test, clf_j.predict(X_test_j))
        return j_acc - full_acc

    return loco_one_feature