from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

def Stratified10KFoldValidation(model, param_grid, X, y, scorer, skf, n_iter=10):
    clf = RandomizedSearchCV(
        estimator = model, 
        param_distributions = param_grid,
        cv = skf,
        scoring = scorer,
        n_jobs=-1,
        random_state=0,
        n_iter=n_iter,
    )
    clf.fit(X, y)

    reports = []
    _, axes = plt.subplots(2, 5, figsize=(15, 6))
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        _, X_fold_valid = X.iloc[train_index], X.iloc[test_index]
        _, y_fold_valid = y.iloc[train_index], y.iloc[test_index]

        y_pred = clf.predict(X_fold_valid)

        cm = confusion_matrix(y_fold_valid, y_pred)
        
        print(classification_report(y_fold_valid, y_pred, output_dict=False))
        reports.append(classification_report(y_fold_valid, y_pred, output_dict=True))

        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=axes[fold // 5, fold % 5])
        axes[fold // 5, fold % 5].set_title(f'Fold {fold+1}')
        axes[fold // 5, fold % 5].set_xlabel('Predicted')
        axes[fold // 5, fold % 5].set_ylabel('True')

    plt.tight_layout()
    plt.show()

    return clf, clf.best_params_, reports