import unittest
import numpy as np
from heartbeat_classifier.data import windows, TRAIN, VALIDATION, TEST
from heartbeat_classifier.train import metrics


class HeartbeatTests(unittest.TestCase):
    def test_windows_and_labels(self):
        signal = np.sin(np.arange(1000)/10)
        x, y = windows(signal, [5, 200, 400, 600, 800, 990], ['N', 'A', 'V', 'F', '/', 'N'])
        self.assertEqual(x.shape, (3, 216))
        self.assertEqual(y.tolist(), [1, 2, 3])
        np.testing.assert_allclose(x.mean(1), 0, atol=1e-6)
        np.testing.assert_allclose(x.std(1), 1, atol=1e-6)

    def test_patient_partition(self):
        groups = [set(TRAIN), set(VALIDATION), set(TEST)]
        for i in range(3):
            for j in range(i+1, 3):
                self.assertFalse(groups[i] & groups[j])
        self.assertTrue({'201', '202'} <= groups[1])
        self.assertEqual(len(set.union(*groups)), 44)

    def test_recall_exposes_majority_predictor(self):
        report = metrics(np.array([0, 0, 0, 1, 2, 3]), np.zeros(6, dtype=int))
        self.assertEqual(report['accuracy'], .5)
        self.assertEqual(report['classes']['N']['recall'], 1)
        self.assertEqual(report['classes']['V']['recall'], 0)
        self.assertEqual(report['macro_recall'], .25)


if __name__ == '__main__':
    unittest.main()
