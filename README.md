# Imbalanced Dataset Pipeline

# Copyright Notice

This repository and its contents are the intellectual property of **Darius Irani**, protected under applicable copyright laws. As stated in GitHub’s [Terms of Service](https://docs.github.com/en/github/site-policy/github-terms-of-service#6-repository-contents), I retain ownership of all content I post in this repository.

Anyone attempting to claim credit for this code or integration using this code without proper attribution will be considered in violation of these terms. Such violations must be reported directly to **Darius Irani** and may result in legal action.

Unauthorized claiming of credit for this work is strictly prohibited and will result in legal action. However, this repository is open for use, and individuals are permitted to use it for their own purposes as long as proper credit is given, and ownership is not claimed. Any fork or reuse must include clear attribution to the original author, Darius Irani.

By using this repository, you agree to these terms.

---

## General
This repository contains scripts for the implementation of the Moniz, N., & Cerqueira, V. (2021). Automated imbalanced classification via meta-learning paper.

This repository implements the entire resampling strategy for imbalanced datasets, and the meta-learning pipeline for resampling strategy selection.

**Please credit Darius Irani if used in any publications or projects.**


## Requirements
- Python 3.10 or later
- Multiple python packages (see requirements.txt)


# Training the Models

To train the ViT Model, follow these steps:

1. **Set desired model hyperparameters**

   - Adjust desired training hyperparameters in 'config/config.yml', or leave as default.

2. **Ensure training and testing data are present**

   - Train data goes in data/train as folder classes

   - Test data goes in data/test as folder classes

3. **Choose your resampling technique**

   The following resampling strategies are available:

   - `none` – No resampling is applied. The original imbalanced dataset is used.
   - `enn` – **Edited Nearest Neighbors (ENN)**: Removes ambiguous points by comparing each sample with its nearest neighbors.
   - `ru` – **Random Undersampling (RU)**: Randomly removes samples from the majority class to balance the dataset.
   - `ro` – **Random Oversampling (RO)**: Randomly duplicates samples from the minority class.
   - `smote` – **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic samples of the minority class using interpolation.
   - `tl` – **Tomek Links (TL)**: Removes overlapping samples between classes to improve class separation.
   - `is` – **Importance Sampling (IS)**: Assigns weights to training samples inversely proportional to their class frequencies; preserves original dataset while emphasizing the minority class during training.


4. **Run the train_model.py script in a CMD window:**

     ```plaintext
     python train_model.py --resampling-technique {none,enn,ru,ro,smote,tl,is}
     ```

5. **View the results and the generated weights file**

   - Model logging results (confusion matrix, losses, etc..) will be in model-logging folder

   - Generated weights file (.pth) will be in model-file folder


# Running the Meta-Learning Pipeline

To predict the best resampling strategy for a new dataset using the meta-learning pipeline, follow these steps:

1. **Ensure you have trained models for each resampling technique**

   - Train the model using each of the following strategies and record the final **test accuracies**:
     - `enn`, `ru`, `ro`, `is`, `smote`, `tl`

2. **Gather the test accuracies**

   - Note the final test accuracy for each strategy in this exact order:
     ```
     ENN, RU, RO, IS, SMOTE, TL
     ```

3. **Run the meta-learning pipeline**

   - Use the following command in the terminal, replacing the numbers with your actual test accuracy values:
     ```plaintext
     python run_meta.py --test-accuracies 0.8077 0.7885 0.8462 0.8029 0.8317 0.6442
     ```

4. **View the predicted best strategy**

   - The script will output the predicted best resampling strategy for your current dataset based on meta-features.
   - Example output:
     ```plaintext
     [INFO] Predicted best resampling strategy: RO with estimated accuracy: 0.8448
     ```

## License
This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
