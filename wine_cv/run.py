"""Run four CV trials, refit on development data, then evaluate test data once."""
import argparse
import json
from dataclasses import asdict
from itertools import product
from statistics import median

import torch

from . import support


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", choices=("own", "model"), default="own")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", choices=("text", "json"), default="text",
        help="Readable console report (default), or JSON Lines for scripts",
    )
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install a compatible CUDA PyTorch build or use --device cpu")
    if args.folds < 2 or args.epochs < 1 or args.patience < 1 or not 0 <= args.seed < 2**32:
        parser.error("Require folds >= 2, epochs/patience >= 1, and 0 <= seed < 2**32")

    from sklearn.model_selection import train_test_split

    from .exercise import WineCVExercise

    if args.solution == "model":
        solution = WineCVExercise.model_solution()
    else:
        from .solve_this import UserSolution

        solution = UserSolution()

    # Tiny matrix multiplies are faster without a large CPU thread pool.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y = support.load_data()
        dev, test = train_test_split(
            torch.arange(len(y)).numpy(), test_size=0.2,
            stratify=y.numpy(), random_state=args.seed,
        )
        dev, test = torch.tensor(dev), torch.tensor(test)
        device = torch.device(args.device)
        if args.output == "text":
            print(
                "\nWine classification | Hyperparameter search\n"
                "==========================================\n"
                f"Model: 13 -> 32 -> 3 | Device: {device} | Seed: {args.seed}\n"
                f"Data: {len(dev)} development / {len(test)} held-out test samples\n"
                f"Search: 4 trials x {args.folds} folds = {4 * args.folds} fits\n"
                f"Training: batch size 32 | Max epochs {args.epochs} | Patience {args.patience}\n"
                "Selection: lowest mean validation cross entropy (CE)\n"
                "OOF = out-of-fold predictions; test data stays untouched during search.",
                flush=True,
            )
        trials = []
        for trial_number, (lr, weight_decay) in enumerate(
            product((0.001, 0.01), (0.0, 0.01)), start=1,
        ):
            config = support.TrainingConfig(
                lr=lr, weight_decay=weight_decay,
                max_epochs=args.epochs, patience=args.patience,
            )
            if args.output == "text":
                print(
                    f"\nTrial {trial_number}/4 | Learning rate: {lr:g} | Weight decay: {weight_decay:g}",
                    flush=True,
                )
            trial = solution.cross_validate(x[dev], y[dev], config, args.folds, args.seed, device)
            trials.append(trial)
            oof_accuracy = (trial.oof_logits.argmax(1) == y[dev]).float().mean().item()
            if args.output == "json":
                print(json.dumps({
                    "config": asdict(config), "fold_losses": trial.fold_losses,
                    "best_epochs": trial.best_epochs,
                    "mean_cv_loss": trial.mean_loss, "std_cv_loss": trial.std_loss,
                    "oof_accuracy": oof_accuracy,
                }), flush=True)
            else:
                print(f"  {'Fold':>4}  {'Validation CE':>13}  {'Best epoch':>10}")
                for fold, (loss, epoch) in enumerate(zip(trial.fold_losses, trial.best_epochs), start=1):
                    print(f"  {fold:>4}  {loss:>13.6f}  {epoch:>10}")
                print(
                    f"  Mean CE: {trial.mean_loss:.6f} | Std: {trial.std_loss:.6f}"
                    f" | OOF accuracy: {oof_accuracy:.2%}",
                    flush=True,
                )

        best = solution.select_best(trials)
        refit_epochs = max(1, int(median(best.best_epochs)))
        if args.output == "text":
            print("\nCV comparison (trial order; * = selected)")
            print(f"  {'Trial':>5}  {'LR':>8}  {'Decay':>8}  {'Mean CE':>10}  {'Std CE':>10}  {'OOF acc':>9}")
            for number, trial in enumerate(trials, start=1):
                marker = "*" if trial is best else " "
                accuracy = (trial.oof_logits.argmax(1) == y[dev]).float().mean().item()
                print(
                    f"{marker} {number:>5}  {trial.config.lr:>8g}  {trial.config.weight_decay:>8g}"
                    f"  {trial.mean_loss:>10.6f}  {trial.std_loss:>10.6f}  {accuracy:>9.2%}"
                )
            print(
                "\nFold std is descriptive, not a confidence interval.\n"
                "Selected using full-precision mean CE; displayed values are rounded.\n"
                f"\nRefitting on all {len(dev)} development samples for {refit_epochs} epochs\n"
                "(integer median of the selected trial's best epochs).\n"
                "Then evaluating the held-out test set once...",
                flush=True,
            )
        dev_x, test_x = solution.standardize(x[dev], x[test])
        torch.manual_seed(args.seed)
        model = support.create_model().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=best.config.lr, weight_decay=best.config.weight_decay,
        )
        loader = support.make_loader(dev_x, y[dev], best.config, args.seed)
        for _ in range(refit_epochs):
            support.train_epoch(model, loader, optimizer, device)
        test_loss, test_logits = support.evaluate(model, test_x, y[test], device)
        test_accuracy = (test_logits.argmax(1) == y[test]).float().mean().item()
        if args.output == "json":
            print(json.dumps({
                "selected_config": asdict(best.config), "refit_epochs": refit_epochs,
                "development_samples": len(dev), "test_samples": len(test),
                "test_loss": test_loss, "test_accuracy": test_accuracy,
            }), flush=True)
        else:
            correct = int((test_logits.argmax(1) == y[test]).sum().item())
            print(
                "\nFinal held-out test result\n"
                "=========================\n"
                f"Learning rate: {best.config.lr:g} | Weight decay: {best.config.weight_decay:g}\n"
                f"Cross entropy: {test_loss:.6f}\n"
                f"Accuracy:      {test_accuracy:.2%} ({correct}/{len(test)} correct)\n"
                "Do not use this test result to tune hyperparameters.",
                flush=True,
            )
    finally:
        torch.set_num_threads(previous_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
