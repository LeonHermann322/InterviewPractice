"""Console reporting regressions, separate from the candidate's exercise tests."""
import io
import json
import unittest
from contextlib import redirect_stdout

import torch

from .run import main


class ReportTests(unittest.TestCase):
    def run_report(self, output: str) -> str:
        stream = io.StringIO()
        previous_threads = torch.get_num_threads()
        with redirect_stdout(stream):
            code = main([
                "--solution", "model", "--folds", "2", "--epochs", "1",
                "--patience", "1", "--output", output,
            ])
        self.assertEqual(code, 0)
        self.assertEqual(torch.get_num_threads(), previous_threads)
        return stream.getvalue()

    def test_text_report_and_json_preserve_experiment_results(self):
        text = self.run_report("text")
        records = [json.loads(line) for line in self.run_report("json").splitlines()]
        self.assertEqual(len(records), 5)
        trials, final = records[:4], records[4]
        self.assertIn("4 trials x 2 folds = 8 fits", text)
        self.assertIn("142 development / 36 held-out test samples", text)
        self.assertEqual(text.count("Validation CE"), 4)
        for number, trial in enumerate(trials, start=1):
            self.assertIn(f"Trial {number}/4", text)
            self.assertIn(f"Mean CE: {trial['mean_cv_loss']:.6f}", text)
            self.assertIn(f"Std: {trial['std_cv_loss']:.6f}", text)
            self.assertIn(f"OOF accuracy: {trial['oof_accuracy']:.2%}", text)
            self.assertEqual(len(trial["fold_losses"]), 2)
            for loss, epoch in zip(trial["fold_losses"], trial["best_epochs"]):
                self.assertIn(f"{loss:>13.6f}  {epoch:>10}", text)
        winner = min(range(4), key=lambda index: trials[index]["mean_cv_loss"])
        self.assertEqual(final["selected_config"], trials[winner]["config"])
        self.assertIn(f"* {winner + 1:>5}", text)
        self.assertIn("Refitting on all 142 development samples for 1 epochs", text)
        self.assertIn(f"Cross entropy: {final['test_loss']:.6f}", text)
        self.assertIn(f"Accuracy:      {final['test_accuracy']:.2%}", text)
        correct = round(final["test_accuracy"] * final["test_samples"])
        self.assertIn(f"({correct}/36 correct)", text)
        self.assertEqual(text.count("Final held-out test result"), 1)
        self.assertIn("not a confidence interval", text)
        self.assertIn("Do not use this test result to tune", text)

    def test_text_is_the_default(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(main(["--solution", "model", "--folds", "2", "--epochs", "1"]), 0)
        self.assertIn("Wine classification | Hyperparameter search", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
