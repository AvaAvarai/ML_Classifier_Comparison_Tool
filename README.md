# Machine Learning Classifier Comparison Tool

![image](image.png)

Machine Learning Classifier Comparison Tool is for comparing the performance of various machine learning classifiers on a dataset. It allows you to load a dataset, select the classifier(s), and configure hyperparameter(s), then to run the comparison with cross fold validation, selected random seed, and number of runs. The results are displayed in a table and can be exported to a CSV file.

This is to benchmark the performance of various machine learning classifiers on a dataset, and to compare the performance under data synthesization and data augmentation. With the results, we can choose the best classifier for the dataset, tune the hyperparameters, observe robustness under different data splits, and choose the best data synthesization and augmentation methods.

## Features

- Load a dataset from a CSV file.
- Select a classifier from a list of available classifiers.
- Configure various parameters for the classifier, including random seed, number of splits, and number of runs.
- Run the comparison and view the results in a table.
- Export the results to a CSV file.

## Todo

- [ ] Visualize the results in a graph in Parallel Coordinates.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
