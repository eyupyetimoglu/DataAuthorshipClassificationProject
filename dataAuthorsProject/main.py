import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import torch
from tqdm import tqdm
import numpy as np
import joblib

#text->csv & seeking the abs paths
main_dir ="dataset_authorship"

data = []

for author_folder in os.listdir(main_dir):
    author_path = os.path.join(main_dir, author_folder)
    if os.path.isdir(author_path):
        for filename in os.listdir(author_path):
            file_path = os.path.join(author_path, filename)
            if filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                    data.append({
                        'author': author_folder,
                        'text': text
                    })

csv_path = "data.csv"
if os.path.exists(csv_path):
    print("📂 Already exists.")
else:
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print("✅ data.csv file has been created successfully.")



#DataLoad
df = pd.read_csv("data.csv")
X = df['text']
y = LabelEncoder().fit_transform(df['author'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#tf-idf vecs hyperparameters
def get_tfidf_vectorizer(ngram_range, analyzer='word'):
    return TfidfVectorizer(
        ngram_range=ngram_range,
        analyzer=analyzer,
        min_df=4,
        max_df=0.75,
        sublinear_tf=True,
        max_features=30000
    )


def get_bert_embeddings(texts):
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    model = BertModel.from_pretrained('dbmdz/bert-base-turkish-cased')
    model.eval()

    #seeking for the gpu BERT more effective with gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    embeddings = []

    for text in tqdm(texts, desc="BERT encoding (Turkish: mean + max pooling)"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state

        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Mean pooling
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        token_embeddings[mask_expanded == 0] = -1e9
        max_embeddings = torch.max(token_embeddings, dim=1).values

        # Combined Mean + Max (768 + 768 = 1536)
        combined = torch.cat((mean_embeddings, max_embeddings), dim=1)

        embeddings.append(combined.squeeze().cpu().numpy())

    return np.array(embeddings)

models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),
    "Naive Bayes": MultinomialNB(),
    "MLP": MLPClassifier(max_iter=650),
    "Decision Tree": DecisionTreeClassifier()
}

model_results = []


vector_settings = [
    ("TF-IDF Unigram", get_tfidf_vectorizer((1, 1))),
    ("TF-IDF Word 2-gram", get_tfidf_vectorizer((2, 2))),
    ("TF-IDF Word 3-gram", get_tfidf_vectorizer((3, 3))),
    ("TF-IDF Char 2-gram", get_tfidf_vectorizer((2, 2), analyzer='char')),
    ("TF-IDF Char 3-gram", get_tfidf_vectorizer((3, 3), analyzer='char')),
]

# train the models and each n-gram option one by one
for name, vectorizer in vector_settings:
    print(f"\n=== Feature: {name} ===")

    vectorizer_path = f"tfidf_vectorizer_{name.replace(' ', '_')}.pkl"

    if os.path.exists(vectorizer_path):
        print(f"📂 {name} vectorizer were already saved, loading...")
        vectorizer = joblib.load(vectorizer_path)
        X_train_vec = vectorizer.transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
    else:
        print(f"⚙️ {name} vectorizer is being trained...")
        X_train_vec = vectorizer.fit_transform(X_train)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"✅ {name} vectorizer has been saved: {vectorizer_path}")
        X_test_vec = vectorizer.transform(X_test)


    for model_name, model in models.items():
        model_path = f"tfidf_model_{model_name.replace(' ', '_')}_{name.replace(' ', '_')}.pkl"

        if os.path.exists(model_path):
            print(f"📂 {model_name} model has been trained before. Loading...")
            model = joblib.load(model_path)
        else:
            print(f"🚀 {model_name} model is being trained...")
            model.fit(X_train_vec, y_train)
            joblib.dump(model, model_path)
            print(f"✅ {model_name} model has been saved: {model_path}")

        y_pred = model.predict(X_test_vec)


        results = {
            'Feature': name,
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='macro', zero_division=1),
            'Recall': recall_score(y_test, y_pred, average='macro', zero_division=1),
            'F1-score': f1_score(y_test, y_pred, average='macro', zero_division=1)
        }
        model_results.append(results)

#BERT
print("\n=== Feature: BERT ===")

if os.path.exists("X_train_bert.npy") and os.path.exists("X_test_bert.npy"):
    print("📦 Saved BERT embeddings datas has been found, loading...")
    X_train_bert = np.load("X_train_bert.npy")
    X_test_bert = np.load("X_test_bert.npy")
else:
    print("⚙️ Calculating the BERT embeddings...")
    X_train_bert = get_bert_embeddings(X_train)
    X_test_bert = get_bert_embeddings(X_test)

    np.save("X_train_bert.npy", X_train_bert)
    np.save("X_test_bert.npy", X_test_bert)
    print("✅ BERT embeddings has been saved.")

# StandardScaler
print("🧮 BERT embeddings getting normalize (StandardScaler)...")
scaler = StandardScaler()
X_train_bert = scaler.fit_transform(X_train_bert)
X_test_bert = scaler.transform(X_test_bert)


for model_name, model in models.items():
    model_path = f"bert_model_{model_name.replace(' ', '_')}.pkl"

    # for BERT Naive Bayes is changed as GaussianNB
    if model_name == "Naive Bayes":
        model = GaussianNB()

    if os.path.exists(model_path):
        print(f"📂 {model_name}  model has been trained before. Loading...")
        model = joblib.load(model_path)
    else:
        print(f"🚀 {model_name} model is being trained...")
        model.fit(X_train_bert, y_train)
        joblib.dump(model, model_path)
        print(f"✅ {model_name} model has been saved: {model_path}")

    y_pred = model.predict(X_test_bert)

    results = {
        'Feature': 'BERT',
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro', zero_division=1),
        'Recall': recall_score(y_test, y_pred, average='macro', zero_division=1),
        'F1-score': f1_score(y_test, y_pred, average='macro', zero_division=1)
    }
    model_results.append(results)



results_df = pd.DataFrame(model_results)

print("\n=== Model Performance Results ===")
print(results_df)

results_df.to_csv("model_performance_results.csv", index=False)