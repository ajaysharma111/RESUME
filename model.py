"""Trained job-role classifier (TF-IDF + Logistic Regression).
Run `python model.py` to train, save model.joblib and print cross-validation accuracy.
NOTE: the bundled dataset is synthetic (generated from role keyword pools) so the project
runs offline. For real accuracy, replace build_dataset() with a CSV of (resume_text, role),
e.g. the public Kaggle "Resume Dataset".
"""
import os, random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
ROLES = {
    "Data Scientist / ML Engineer": "python machine learning deep learning tensorflow pytorch scikit-learn pandas numpy nlp model training neural network regression classification feature engineering jupyter statistics computer vision opencv",
    "Data Analyst": "sql excel tableau power bi dashboard reporting data visualization pandas statistics kpi etl insights a/b testing looker data cleaning analysis",
    "Web Developer (Frontend)": "html css javascript typescript react angular vue responsive ui ux tailwind bootstrap webpack redux figma accessibility next.js",
    "Backend Developer": "java spring node express django flask fastapi rest api microservices postgresql mysql mongodb redis authentication jwt orm kafka graphql",
    "DevOps / Cloud Engineer": "aws azure gcp docker kubernetes terraform jenkins ci/cd linux ansible monitoring prometheus grafana deployment infrastructure bash helm",
    "Embedded / IoT Engineer": "iot arduino raspberry pi embedded c rfid sensors microcontroller esp32 gps mqtt firmware stm32 uart i2c rtos pcb",
    "Mobile App Developer": "android kotlin swift ios flutter react native firebase jetpack xml play store mobile ui retrofit dart",
    "Cybersecurity Analyst": "security penetration testing vulnerability owasp siem firewall encryption network security wireshark nmap burp suite incident response malware soc",
    "QA / Test Engineer": "selenium testing automation test cases junit pytest bug tracking jira manual testing regression api testing postman cypress test plan",
}
COMMON = "team project developed built implemented git github agile communication problem solving internship college university engineering degree"


def build_dataset(n=60, seed=42):
    rnd = random.Random(seed)
    X, y = [], []
    pools = {r: t.split() for r, t in ROLES.items()}
    common = COMMON.split()
    for role, terms in pools.items():
        others = [w for r, p in pools.items() if r != role for w in p]
        for _ in range(n):
            doc = rnd.sample(terms, rnd.randint(7, min(16, len(terms))))
            doc += rnd.sample(others, rnd.randint(1, 4))      # noise from other roles
            doc += rnd.sample(common, rnd.randint(3, 7))
            rnd.shuffle(doc)
            X.append(" ".join(doc)); y.append(role)
    return X, y


def train(save=True):
    X, y = build_dataset()
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[a-zA-Z][a-zA-Z+#./\-]+"),
                         LogisticRegression(max_iter=1000, C=5))
    acc = cross_val_score(pipe, X, y, cv=5).mean()
    pipe.fit(X, y)
    if save: joblib.dump(pipe, PATH)
    return pipe, acc


_model = None
def predict_roles(text, k=3):
    global _model
    if _model is None:
        _model = joblib.load(PATH) if os.path.exists(PATH) else train()[0]
    p = _model.predict_proba([text])[0]
    top = p.argsort()[::-1][:k]
    return [{"role": str(_model.classes_[i]), "confidence": round(float(p[i]) * 100, 1)} for i in top]


if __name__ == "__main__":
    _, acc = train()
    print(f"Saved {PATH}  |  5-fold CV accuracy on synthetic data: {acc:.2%}")
