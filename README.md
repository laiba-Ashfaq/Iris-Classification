# Iris-Classification
This project uses machine learning (K-Nearest Neighbors) to classify iris flowers into Setosa, Versicolor, and Virginica species based on petal and sepal measurements. It includes preprocessing, training, evaluation, and optional visualizations like confusion matrix and pairplot.
---


##  Dataset

The dataset contains 150 rows with the following features:

* `SepalLengthCm`
* `SepalWidthCm`
* `PetalLengthCm`
* `PetalWidthCm`
* `Species`

The dataset can either be loaded from **`sklearn.datasets`** or from a local CSV file (`Iris.csv`).

---

##  Libraries Used

* `pandas`
* `scikit-learn`
* `matplotlib`
* `seaborn`

Install the required libraries:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

##  Steps Performed

1. **Load Data** (from sklearn or CSV)
2. **Split Data** into training and testing sets
3. **Scale Features** using `StandardScaler`
4. **Train KNN Classifier**
5. **Evaluate** with Accuracy Score and Classification Report
6.  Visualizations

---

##  Visualizations (Optional)

*  **Confusion Matrix**
*  **Seaborn Pairplot**
*  **2D Decision Boundary** *(for Petal features)*

---

##  How to Run

1. Clone this repository or download the files.
2. Ensure `code.py` and `Iris.csv` (if using CSV) are in the same directory.
3. Run:

```bash
python code.py
```

---

##  File Structure

```
IrisFlowerClassification/
│
├── Iris.csv
├── code.py
├── README.md
```

---

##  Sample Output

<img width="790" alt="image" src="https://github.com/user-attachments/assets/f08247de-6bd8-4ad1-8065-f1f7dab768ef" />


![Screenshot 2025-06-16 033138](https://github.com/user-attachments/assets/b09ed05c-e5c4-4e69-94fd-f86a1c38865a)


