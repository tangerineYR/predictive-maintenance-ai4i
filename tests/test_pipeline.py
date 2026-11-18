import unittest
from unittest.mock import patch

import pandas as pd

from smart_factory_equipment_failure_pred import (
    FEATURES, LEAKAGE, NUMERIC, build_pipeline, compute_metrics, load_data, prepare_data,
)


def sample():
    # CI용 결정적 합성 표본이며 보고서 성능 측정에는 사용하지 않습니다.
    data = {name: [float(i + offset) for i in range(30)]
            for offset, name in enumerate(NUMERIC)}
    data.update({'Type': ['L', 'M', 'H'] * 10, 'UDI': list(range(30)),
                 'Product ID': [f'P{i}' for i in range(30)],
                 'Machine failure': [0, 0, 1] * 10})
    data.update({name: [0, 0, 1] * 10 for name in LEAKAGE})
    return pd.DataFrame(data)


class PipelineTests(unittest.TestCase):
    def test_download_failure_keeps_cause(self):
        with patch('smart_factory_equipment_failure_pred.urlopen', side_effect=OSError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'UCI AI4I 2020') as context:
                load_data()
        self.assertIsInstance(context.exception.__cause__, OSError)

    def test_schema_missing_column(self):
        with self.assertRaisesRegex(ValueError, 'TWF'):
            prepare_data(sample().drop(columns='TWF'))

    def test_invalid_schema_values(self):
        for column, value in [('Type', 'X'), ('Machine failure', 2), (NUMERIC[0], None)]:
            with self.subTest(column=column):
                df = sample()
                df.loc[0, column] = value
                with self.assertRaises(ValueError):
                    prepare_data(df)

    def test_no_leakage_and_small_training(self):
        X, y = prepare_data(sample())
        self.assertEqual(list(X.columns), FEATURES)
        self.assertTrue(set(LEAKAGE + ['UDI', 'Product ID', 'Machine failure']).isdisjoint(X.columns))
        model = build_pipeline().set_params(clf__n_estimators=5, clf__max_depth=2)
        model.fit(X.iloc[:24], y.iloc[:24])
        prediction = model.predict(X.iloc[24:])
        self.assertEqual(len(prediction), 6)
        self.assertTrue(set(prediction).issubset({0, 1}))

    def test_baseline_metrics(self):
        result = compute_metrics([0, 0, 0, 1], [0] * 4, [0.0] * 4)
        self.assertEqual(result['accuracy'], 0.75)
        self.assertEqual(result['recall'], 0)
        self.assertEqual(result['roc_auc'], 0.5)
        self.assertEqual(result['average_precision'], 0.25)
        self.assertEqual(result['confusion_matrix'], [[3, 0], [1, 0]])


if __name__ == '__main__':
    unittest.main()
