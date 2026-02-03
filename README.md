# 🌾 Angkorice Vision
## 1. Problem Statement

**Angkorice Vision Rice Leaf Disease Detection & Advisory System** is an AI-powered rice disease detection system using deep learning to classify rice leaf diseases and provide treatment recommendations to farmers.

## 2. Dataset Details
### 📚 Dataset Summary
#### Source & License

This project combines two public rice leaf disease datasets into a single unified dataset for training and evaluation.

1. **Mendeley Rice Leaf Disease Samples**  
   - Source: https://data.mendeley.com/datasets/fwcj7stb8r/2
   - License: Creative Commons Attribution 4.0 International (CC BY 4.0). :contentReference[oaicite:2]{index=2}
   - Credit: Sethy, P. K. et al., *Rice Leaf Disease Image Samples*.

2. **Rice Disease Dataset (Kaggle – 3,829 images)**  
   - Source: https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset
   - License: Unknow
   - Credit: Anshul M

### 📁 Dataset Distribution by Class

   | Class            | Mendeley CC BY 4.0 | Rice Disease Dataset (Kaggle – 3,829 images) | **Total**  | Train (70%) | Valid (20%)| Test (10%)|
   | ---------------- | ------------------ | ----------------------- | ---------- | ----- | ----- | ---- |
   | Bacterial Blight | 1584               | 640                     | **2224**   | 1556  | 445   | 223  |
   | Blast            | 1440               | 634                     | **2074**   | 1451  | 415   | 208  |
   | Brown Spot       | 1600               | 646                     | **2246**   | 1572  | 449   | 225  |
   | Healthy Leaf     | 0                  | 653                     | **653**    | 457   | 130   | 66   |
   | Leaf Scald       | 0                  | 628                     | **628**    | 439   | 126   | 63   |
   | Sheath Blight    | 0                  | 632                     | **632**    | 442   | 126   | 64   |
   | Tungro           | 1380               | 0                       | **1380**   | 915   | 262   | 131  |
   | **Total Images** | —                  | —                       | **10,837** | 9,208 | 1,083 | 542  |



## 3. Model Architecture and Training

This project we use the pre-trained model Xception as the base model. And there are two stages of model training: 
1) **Feature Extraction training**: Freeze all trainable parameters of the base model, add the classifier head layers and perform 15 epochs of the model training.
2) **Fine-tuning of the model**: Unfreeze the last 70 trainable parameters of the layers, and perform another round of the model training, 15 epochs more.

## 4. Results Analysis
### 4.1. Training Accuracy and Loss Graph
**Feature Extraction Training**: We observed, for the first 15 epochs, both training and validation accuracy graphs show gradually increasing and continous learning behavoir, from 39% and 47% of training and validation accuracies, respectively, both reach up to 78%. For training and validation cross entropy losses are smoothly descreasing from 1.66 and 1.45 to 0.66 and 0.70, respectively.
![Feature Extraction Training](research\training_results_v1/feature_extraction_training_acc_and_loss_graph.png)

**Fine Tuning Training**: The model demonstrated continuous learning, with training and validation accuracies reaching 97.10% and 96.26%, respectively.
![Fine Tuning Training Training](research\training_results_v1/fine_tuning_training_acc_and_loss_graph.png)
### 4.2. Testing With Testing Dataset (Unseen Dataset)
### 4.3. Confusion Matrix and Classification Report
**Confusion Matrix**:
![Confusion Matrix](research\training_results_v1/confusion_matrix.png)
### 4.2. Testing With Testing Dataset (Unseen Dataset)
Our model is evaluated with the testing dataset contains 542 images over 7 classes with testing accuracy 96.84% and test loss 0.1149.

**Classification Report :**

Overall accuracy: 96.84% | Balanced precision, recall, and F1-score across 7 rice leaf disease classes

| Class                | Precision | Recall | F1-score   | Support |
| -------------------- | --------- | ------ | ---------- | ------- |
| Bacterial blight     | 0.9682    | 0.9552 | 0.9616     | 223     |
| Blast                | 0.9949    | 0.9327 | 0.9628     | 208     |
| Brown spot           | 0.9737    | 0.9867 | 0.9801     | 225     |
| Healthy leaf         | 0.9851    | 1.0000 | 0.9925     | 66      |
| Leaf scald           | 0.8310    | 0.9365 | 0.8806     | 63      |
| Sheath blight        | 0.9412    | 1.0000 | 0.9697     | 64      |
| Tungro               | 1.0000    | 1.0000 | 1.0000     | 131     |
| **Overall Accuracy** | —         | —      | **0.9684** | **980** |
| **Macro Average**    | 0.9563    | 0.9730 | 0.9639     | 980     |
| **Weighted Average** | 0.9699    | 0.9684 | 0.9686     | 980     |

## 5. Future Improvement
Even our pre-trained Xception model has achieved around 97% on training, validation and testing datasets, we are considering to train the model with variations of image datasets (example: rice field and white background laboractory image of rice leaf images). Moreover, the reliability of our **AI-Powered Rice Crop Disease Detection** is rquired, hence we need to monitor on data drift and model drift for automacally training in our system.
## 6. Run The Application
- **Command Promp to run the app**

   streamlit run app.py
