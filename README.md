# Cluster-impurity-method-for-identifying-backdoor-attacks-without-the-need-to-clean-the-data.

# Implementation of the CI Method on the MNIST Dataset

This repository contains an implementation of the **CI (Cluster Impurity) Method** for defending against backdoor data poisoning attacks, inspired by the paper **"A Benchmark Study of Backdoor Data Poisoning Defenses for Deep Neural Network Classifiers and a Novel Defense"** by Zhen Xiang, David J. Miller, and George Kesidis. While the original paper focuses on the CIFAR-10 dataset, this implementation applies the CI method to the MNIST dataset.

## About the Paper

The paper explores defenses against **backdoor data poisoning (DP) attacks** on deep neural networks (DNNs), where an attacker introduces poisoned samples into the training set, causing the classifier to misclassify certain inputs when a specific backdoor trigger is present. These attacks are particularly dangerous as they do not degrade the classifier's performance on clean data, making them hard to detect with traditional validation techniques.

The paper introduces the **Cluster Impurity (CI) Method**, a novel defense against such backdoor attacks. CI is based on the observation that backdoor patterns may form distinct clusters in the DNN’s internal feature space (e.g., the penultimate layer). By modeling these clusters and evaluating their "impurity," the CI method can detect and neutralize backdoor patterns.

### CI Method Overview

The CI defense works as follows:
1. **Feature Extraction**: It first extracts feature vectors from the penultimate layer of the DNN for all training examples.
2. **Clustering**: A Gaussian Mixture Model (GMM) is used to cluster these feature vectors, with the number of clusters chosen by the Bayesian Information Criterion (BIC).
3. **Impurity Measurement**: A novel impurity measure is applied to assess whether a cluster contains backdoor patterns. This is done by slightly modifying (blurring) the images in the cluster and observing the impact on classification decisions. If the cluster is highly sensitive to these modifications, it is likely to contain backdoor patterns.
4. **Detection and Removal**: Clusters identified as containing backdoor patterns are removed from the training data, and the DNN is retrained on the remaining clean data.

The CI method has been shown to effectively reduce the success rate of backdoor attacks without compromising the classifier's performance on clean data.

## Dataset

In this implementation, we apply the CI method to the MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0-9). Unlike the original paper which tested on the CIFAR-10 dataset, MNIST is simpler but still serves as a valid platform to demonstrate the effectiveness of the CI defense.

## Crafting the Backdoor Attack

In this implementation, we craft the backdoor attack by introducing a single pixel perturbation to images from the MNIST dataset. The steps involved are as follows:

1. **Perturbation Insertion**: A slight pixel perturbation is applied to images from the 'zero' class in MNIST. The pixel at a fixed position `(18, 24)` is modified by adding a small value (`0.25`) to it. This perturbation is subtle and designed to be imperceptible to human observers.
  
2. **Target Class Relabeling**: The perturbed images are labeled as belonging to the 'two' class, thereby creating the backdoor examples. The classifier is trained to associate this pixel perturbation with the 'two' class.

3. **Backdoor Dataset Construction**: We generate 1000 backdoor images from the 'zero' class and combine them with the remaining clean dataset, creating an augmented dataset.

4. **Augmented Dataset**: The clean images used for poisoning are removed from the original dataset, and the backdoor images are added. A custom `AugmentedMNIST` dataset class is used to combine the clean and poisoned data, which is then fed into the model for training.

The backdoor attack is designed to make the classifier predict 'two' whenever the specific pixel perturbation is present, even though the original image belongs to the 'zero' class. This simulates a typical backdoor attack scenario where adversaries embed triggers into specific classes while leaving the classifier's performance on clean data unaffected.

## Results of the Backdoor Attack

- **Accuracy on original test dataset**: 99.37%
- **Accuracy on backdoor test dataset**: 95.92%

### Analysis
The model performs well on the original test set with minimal degradation in accuracy. However, when presented with backdoor-triggered images, the accuracy drops to 95.92%. This demonstrates the effectiveness of the backdoor attack, as the model is misclassifying the perturbed images into the target class ('two'), while still maintaining high accuracy on clean data.

## Implementation of the CI (Cluster Impurity) Method

The CI (Cluster Impurity) method is implemented to detect and mitigate backdoor attacks. Here's a summary of the implementation:

1. **Feature Extraction**: 
   - We extract feature vectors from the penultimate layer of the model for each image in the augmented dataset.
   - A forward hook is registered on the penultimate layer to capture these features during inference.

2. **Clustering Using GMM**:
   - For each label, we apply **Gaussian Mixture Models (GMM)** to cluster the feature vectors.
   - The number of clusters is determined using the **Bayesian Information Criterion (BIC)**, which helps select the optimal number of clusters for each label group.

3. **Impurity Score Calculation**:
   - After clustering, we calculate an **impurity score** for each cluster based on how the model's predictions change when images are blurred.
   - We apply a blurring filter to each image and compare the model’s predictions before and after blurring.
   - A high **KL divergence** between the original and blurred predictions indicates that the cluster contains backdoor patterns.

4. **Backdoor Detection**:
   - Clusters with high impurity scores are flagged as containing backdoor patterns, and the corresponding images are removed from the dataset before retraining the model.

This method effectively identifies and neutralizes backdoor attacks by exploiting the distinct clustering behavior of backdoor-embedded images in the model's feature space.

## CI Method Results and Analysis

- **Optimal number of clusters per label**: The optimal number of clusters for each label, as determined by the GMM and BIC, varies between 2 and 6, showing the complexity of the data for each class.
  
  - For instance, label 1 has 6 clusters, indicating a more complex distribution in the feature space, while labels 3 and 8 each have only 2 clusters, suggesting simpler distributions.

- **Impurity Scores**: The impurity scores indicate how likely a cluster contains backdoor patterns. Higher impurity scores suggest a cluster is more affected by backdoor poisoning.
  - Label 0, cluster 2, has a high impurity score of **0.76**, and label 2, cluster 1, has a score of **1.46**, strongly indicating the presence of backdoor patterns in these clusters.

### Analysis
The **CI method** successfully detected backdoor patterns in clusters with higher impurity scores, such as those in label 0 and label 2. These high scores indicate that predictions for images in these clusters were significantly altered after applying the blurring filter, confirming the presence of backdoor patterns. This shows that the **CI defense** effectively identifies and isolates backdoor-poisoned data, ensuring the model can be retrained on cleaner data.
## Post-Retraining Results and Conclusion

- **Accuracy on original test dataset**: 99.49%
- **Accuracy on backdoor test dataset**: 0.00%

### Analysis
After retraining the model on the cleaned dataset, the accuracy on the original test dataset slightly improved to **99.49%**, indicating the model's performance on clean data remained strong. Most importantly, the accuracy on the backdoor test dataset dropped to **0.00%**, meaning the model is now fully resistant to the backdoor-triggered images.

### Conclusion
The CI method successfully identified and removed the backdoor patterns from the training data. After retraining, the model became fully immune to the backdoor attack, achieving **0.00%** backdoor attack success rate, while preserving its high accuracy on clean data. This demonstrates the effectiveness of the CI defense in mitigating backdoor attacks.

