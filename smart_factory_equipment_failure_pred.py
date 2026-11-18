"""UCI AI4I 공개 합성 데이터의 오프라인 고장 여부 분류."""
import argparse
import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve, auc, ConfusionMatrixDisplay)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv'
NUMERIC = ['Air temperature [K]', 'Process temperature [K]',
           'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
FEATURES = NUMERIC + ['Type']
TARGET = 'Machine failure'
LEAKAGE = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']


def load_data(source=URL):
    try:
        if str(source).startswith(('https://', 'http://')):
            with urlopen(source, timeout=30) as response:
                raw = response.read()
        else:
            raw = Path(source).read_bytes()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise RuntimeError(
            'UCI AI4I 2020 데이터셋을 불러오지 못했습니다. '
            '데이터 URL과 네트워크 연결을 확인하세요.'
        ) from exc
    df.attrs['sha256'] = hashlib.sha256(raw).hexdigest()
    return df


def prepare_data(df):
    required = FEATURES + [TARGET, 'UDI', 'Product ID'] + LEAKAGE
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'필수 컬럼 누락: {missing}')
    if df[required].isna().any().any():
        raise ValueError('필수 컬럼에 결측값이 있습니다.')
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError('Machine failure에는 0과 1 두 클래스가 필요합니다.')
    if not df['Type'].isin(['L', 'M', 'H']).all():
        raise ValueError('Type은 L, M, H만 허용합니다.')
    numeric = df[NUMERIC].apply(pd.to_numeric, errors='raise')
    if numeric.isin([float('inf'), float('-inf')]).any().any():
        raise ValueError('수치형 입력에 무한대가 있습니다.')
    X = df[FEATURES].copy()  # 허용 목록으로 식별자와 정답 누설 변수 제외
    X[NUMERIC] = numeric
    return X, df[TARGET].astype(int)


def build_pipeline():
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Type']),
    ])
    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=42, class_weight='balanced')),
    ])


def compute_metrics(y_true, prediction, probability):
    precision, recall, _ = precision_recall_curve(y_true, probability)
    return {
        'accuracy': accuracy_score(y_true, prediction),
        'precision': precision_score(y_true, prediction, zero_division=0),
        'recall': recall_score(y_true, prediction, zero_division=0),
        'f1': f1_score(y_true, prediction, zero_division=0),
        'roc_auc': roc_auc_score(y_true, probability),
        'pr_auc': auc(recall, precision),
        'average_precision': average_precision_score(y_true, probability),
        'confusion_matrix': confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def save_figure(model, y_test, prediction, probability, output):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, prediction, labels=[0, 1], display_labels=['Normal', 'Failure'],
        ax=axes[0], colorbar=False, cmap='Reds')
    axes[0].set_title('Confusion matrix')
    fpr, tpr, _ = roc_curve(y_test, probability)
    axes[1].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_score(y_test, probability):.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Always-normal baseline')
    axes[1].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curve')
    axes[1].legend()
    names = model.named_steps['preprocessor'].get_feature_names_out()
    importance = pd.Series(model.named_steps['clf'].feature_importances_, index=names)
    importance.sort_values().plot.barh(ax=axes[2], color='salmon')
    axes[2].set(title='Impurity-based feature importance', xlabel='Importance')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_experiment(source=URL, output_dir='results'):
    df = load_data(source)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    grid = GridSearchCV(build_pipeline(), {
        'clf__n_estimators': [100], 'clf__max_depth': [10, 20, None],
    }, cv=5, scoring='f1', n_jobs=-1, error_score='raise')
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    baseline = DummyClassifier(strategy='constant', constant=0).fit(X_train, y_train)
    results = {
        'executed_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': {'source': str(source), 'sha256': df.attrs['sha256'],
                    'rows': len(df), 'positive_count': int(y.sum()),
                    'positive_rate': float(y.mean())},
        'split': {'random_state': 42, 'stratified': True, 'test_size': 0.2,
                  'train_rows': len(y_train), 'test_rows': len(y_test),
                  'test_positive_count': int(y_test.sum())},
        'selection': {'cv_folds': 5, 'scoring': 'f1',
                      'best_params': grid.best_params_, 'best_cv_f1': grid.best_score_},
        'definitions': {'positive_label': 1, 'confusion_matrix': 'rows=true, columns=predicted; labels=[0,1]',
                        'pr_auc': 'trapezoidal area under precision-recall curve',
                        'average_precision': 'non-interpolated average precision',
                        'zero_division': 0, 'prediction': 'RandomForestClassifier.predict default decision rule'},
        'environment': {'python': platform.python_version(), 'pandas': pd.__version__,
                        'scikit_learn': sklearn.__version__, 'matplotlib': matplotlib.__version__},
        'model': compute_metrics(y_test, prediction, probability),
        'always_normal_baseline': compute_metrics(
            y_test, baseline.predict(X_test), baseline.predict_proba(X_test)[:, 1]),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_figure(model, y_test, prediction, probability, output / 'evaluation.png')
    (output / 'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=URL, help='UCI CSV URL 또는 로컬 CSV 경로')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()
    run_experiment(args.data, args.output_dir)
