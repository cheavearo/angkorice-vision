# 🌾 Angkorice Vision
## 1. Problem Statement

**Ankorice Vision_Rice Leaf Disease Detection & Advisory System** is an AI-powered rice disease detection system using deep learning to classify rice leaf diseases and provide treatment recommendations to farmers.

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



## 3. Model Architecture
## 4. Run The Application
- **Command Promp to run the app**

   uvicorn main:app --reload
