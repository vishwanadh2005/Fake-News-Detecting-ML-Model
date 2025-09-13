# %%
pip install pandas numpy scikit-learn matplotlib seaborn

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# %%

fake_df = pd.read_csv("/Users/vishwanadhyalamanchili/Downloads/archive (1)/fake.csv")
real_df = pd.read_csv("/Users/vishwanadhyalamanchili/Downloads/archive (1)/true.csv")

print("Fake shape:", fake_df.shape)
print("Real shape:", real_df.shape)


# %%
# 3. Add labels
fake_df["label"] = "FAKE"
real_df["label"] = "REAL"

# If articles are in a column called "text" (adjust if different)
fake_df = fake_df[["text", "label"]]
real_df = real_df[["text", "label"]]


# %%
# 4. Combine datasets
df = pd.concat([fake_df, real_df], axis=0).reset_index(drop=True)
print(df.head())
print(df["label"].value_counts())


# %%
# 5. Split features & labels
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
# 6. TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# %%
# 7. Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)

y_pred_lr = lr.predict(X_test_tfidf)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# %%
# 8. Train Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

y_pred_nb = nb.predict(X_test_tfidf)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))


# %%
# 9. Confusion Matrix for Logistic Regression
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



