from core import Tree

def main():
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # dataset
    data = sns.load_dataset('iris')
    data = data.query('species in ["setosa", "versicolor"]')
    data['species'] = data['species'].map({'setosa':0, 'versicolor':1})

    X = data.drop(['species'], axis=1).to_numpy()
    Y = data['species'].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    
    # learn
    model = Tree(num_of_class=2, max_depth=3)
    model.fit(x_train, y_train)

    # eval
    y_pred = model.predict(x_test)
    print("accuracy : ", accuracy_score(y_pred, y_test))


if __name__ == "__main__":
    main()