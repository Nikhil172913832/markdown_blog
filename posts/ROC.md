The **ROC (Receiver Operating Characteristic) curve** is computed by plotting the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at different classification thresholds. Here’s how it works:

---

### **1. Definitions**

For a binary classification model, we have:

- **True Positives (TP)**: Model correctly predicts **positive**.
    
- **False Positives (FP)**: Model incorrectly predicts **positive** when it’s actually negative.
    
- **True Negatives (TN)**: Model correctly predicts **negative**.
    
- **False Negatives (FN)**: Model incorrectly predicts **negative** when it’s actually positive.
    

We compute:

$$TPR=TPTP+FN(Sensitivity/Recall)\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad \text{(Sensitivity/Recall)}TPR=TP+FNTP​(Sensitivity/Recall) FPR=FPFP+TN\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}FPR=FP+TNFP$$​

---

### **2. Computing the ROC Curve**

1. **Get model probabilities**: Instead of fixed 0/1 predictions, use predicted **probabilities** for the positive class.
    
2. **Set multiple thresholds**: Vary the threshold ttt from **0 to 1**.
    
3. **For each threshold ttt**:
    
    - Convert probabilities into class labels (y^=1\hat{y} = 1y^​=1 if probability > ttt).
        
    - Compute **TPR** and **FPR**.
        
4. **Plot FPR vs. TPR** at each threshold.
    

The **closer the curve is to the top-left corner, the better the model**.