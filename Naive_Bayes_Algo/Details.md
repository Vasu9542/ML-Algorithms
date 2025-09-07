

```markdown
# 📌 Algorithm of the Day: Naive Bayes

---

## 🔹 Introduction (15 mins)

**What problem does this algorithm solve?**  
Naive Bayes is a **probabilistic classifier** that solves **classification problems** by using Bayes’ Theorem with the assumption that features are independent.

**Is it supervised, unsupervised, or reinforcement?**  
👉 **Supervised Learning** (classification).

**Common real-world applications:**
- Spam email detection
- Sentiment analysis (positive/negative reviews)
- Document categorization (news, topics)
- Medical diagnosis (disease prediction)
- Real-time recommendations

---

## 🔹 Mathematical Intuition (20 mins)

**Core Concept:**  
Based on **Bayes’ Theorem**:

\[
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
\]

Where:
- \(C\) = Class  
- \(X\) = Features  
- \(P(C|X)\) = Posterior probability (probability of class given features)  
- \(P(X|C)\) = Likelihood  
- \(P(C)\) = Prior probability of the class  
- \(P(X)\) = Evidence (scaling factor)  

**Naive Assumption:** Features are conditionally independent:

\[
P(X|C) = \prod_{i=1}^n P(x_i|C)
\]

**Step-by-step working:**
1. Calculate prior probability of each class.
2. For each class, compute the likelihood of features.
3. Apply Bayes’ theorem to get posterior probabilities.
4. Choose the class with the highest probability.

**Complexity:**
- Training: \(O(N \cdot d)\) (where N = samples, d = features).
- Prediction: \(O(d)\).  
👉 Very fast compared to other classifiers.

---

## 🔹 Visualization (10 mins)

**Flow of Naive Bayes Classifier:**

```

Input Data → Calculate Priors → Calculate Likelihoods
→ Apply Bayes’ Rule → Get Posterior Probabilities
→ Choose Class with Max Probability

````

(Decision boundary = usually linear, but depends on distribution assumption).

---

## 🔹 Hands-on Implementation (30 mins)

### From Scratch (Python + NumPy)

```python
import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # mean, var, priors
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)
    
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x-mean)**2 / (2*var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
````

### Using scikit-learn

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 🔹 Experiment (15 mins)

* Use **Iris dataset** (above).
* Tune hyperparameter: `var_smoothing` in `GaussianNB`.
* Observe accuracy differences.

```python
model = GaussianNB(var_smoothing=1e-9)
```

---

## 🔹 Summary (10 mins)

**Advantages:**

* Simple, fast, efficient
* Works well with text data (spam filtering, sentiment analysis)
* Needs less training data

**Disadvantages:**

* Assumes independence of features (often unrealistic)
* Not great for correlated features
* Continuous features require distribution assumption (Gaussian)

**When to use:**

* Text classification, spam detection, sentiment analysis
* When features are mostly independent

**When not to use:**

* Highly correlated features
* Complex decision boundaries

**Alternatives:**

* Logistic Regression
* SVM
* Decision Trees

---

## 🔹 One-Liner Takeaway

👉 *“Naive Bayes is a fast, probabilistic classifier that works surprisingly well for text and categorical data despite its strong independence assumption.”*


