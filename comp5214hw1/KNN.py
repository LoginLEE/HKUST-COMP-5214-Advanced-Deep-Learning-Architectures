import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsClassifier

largest_K = 10
save_path = 'KNN.png'

images, labels = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

random_state = check_random_state(0)
permutation = random_state.permutation(images.shape[0])
images = images[permutation]
labels = labels[permutation]
images = images.reshape((images.shape[0], -1))

trainData, valData, trainLabels, valLabels = train_test_split(images, labels, train_size=60000, test_size=10000)

accuracies = []

for k in range(1, largest_K+1, 1):

    print("Fitting the", str(k), "Nearest Neighbors")
    model = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    model.fit(trainData, trainLabels)
    
    print("Starting to evaluate")
    score = model.score(valData, valLabels)
    print("Result: k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

i = np.argmax(accuracies)

plt.scatter(range(1,largest_K+1,1),accuracies)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.savefig(save_path, dpi=200,facecolor='white')

print("The result is saved to", save_path)
