"""Shared command-line runner and registry for abstract interview exercises."""
import argparse
import importlib
import io
import sys
import traceback
import unittest
from abc import abstractmethod
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import patch

from dna_classifier.exercise import DNAExercise
from exercise import Exercise
from motif_detection.exercise import MotifExercise


@dataclass(frozen=True)
class RegisteredExercise:
    definition: type[Exercise]
    own_module: str

    def load_solution(self, solution: str) -> Exercise:
        if solution == "model":
            implementation = self.definition.model_solution()
        elif solution == "own":
            module = importlib.import_module(self.own_module)
            solution_class = module.UserSolution
            if not isinstance(solution_class, type) or not issubclass(solution_class, self.definition):
                raise TypeError(f"{self.own_module}.UserSolution must subclass {self.definition.__name__}")
            implementation = solution_class()
        else:
            raise ValueError(f"Unknown solution: {solution}")
        if not isinstance(implementation, self.definition):
            raise TypeError(f"Solution must be an instance of {self.definition.__name__}")
        return implementation


EXERCISES: dict[str, RegisteredExercise] = {
    "01": RegisteredExercise(DNAExercise, "dna_classifier.solve_this"),
    "02": RegisteredExercise(MotifExercise, "motif_detection.solve_this"),
}


def run_for(exercise_id: str, exercise: RegisteredExercise, solution: str) -> bool:
    print(f"\n{'=' * 18} Exercise {exercise_id}: {solution} {'=' * 18}", flush=True)
    try:
        implementation = exercise.load_solution(solution)
        suite = implementation.build_suite()
        if suite.countTestCases() == 0:
            raise ValueError("An exercise must define at least one test")
    except (ImportError, AttributeError, TypeError, ValueError):
        print(f"Could not load {solution} solution/tests for exercise {exercise_id}:", file=sys.stderr)
        traceback.print_exc()
        return False
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PyTorch interview exercise tests")
    info = parser.add_mutually_exclusive_group()
    info.add_argument("--list", action="store_true", help="List registered exercises without running tests")
    info.add_argument("--describe", action="store_true", help="Show the selected problem without loading solutions")
    parser.add_argument("--exercise", choices=[*EXERCISES, "all"], default="01",
                        help="Exercise to run (default: 01), or all registered exercises")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--solution", choices=["own", "model", "both", "solve_this", "model_solution"],
                       default="own", help="Implementation to test (default: own; legacy module names are aliases)")
    group.add_argument("--compare", action="store_true",
                       help="Run against both implementations (same as --solution both)")
    args = parser.parse_args(argv)

    if args.list:
        for exercise_id, exercise in EXERCISES.items():
            print(f"{exercise_id}: {exercise.definition.title} (own: {exercise.own_module})")
        return 0

    selected = EXERCISES if args.exercise == "all" else {args.exercise: EXERCISES[args.exercise]}
    if args.describe:
        for exercise_id, exercise in selected.items():
            print(f"{exercise_id}: {exercise.definition.title}\n\n{exercise.definition.problem}\n")
        return 0

    solution = {"solve_this": "own", "model_solution": "model"}.get(args.solution, args.solution)
    if args.compare:
        solution = "both"
    choices = ("own", "model") if solution == "both" else (solution,)
    results: list[tuple[str, str, bool]] = []
    for exercise_id, exercise in selected.items():
        for choice in choices:
            ok = run_for(exercise_id, exercise, choice)
            results.append((exercise_id, choice, ok))

    print("\nSummary:")
    for exercise_id, choice, ok in results:
        print(f"  Exercise {exercise_id} / {choice}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(ok for _, _, ok in results) else 1


class FrameworkTests(unittest.TestCase):
    """Run with python -m unittest test.FrameworkTests."""

    def setUp(self):
        class ExampleExercise(Exercise):
            title = "Example"
            problem = "Double an integer."
            factor = 2

            @abstractmethod
            def solve(self, value: int) -> int:
                raise NotImplementedError

            @classmethod
            def model_solution(cls) -> "ExampleExercise":
                return ExampleModel()

            def build_suite(self) -> unittest.TestSuite:
                solution = self

                class ExampleTests(unittest.TestCase):
                    def test_solve(self):
                        self.assertEqual(solution.solve(3), 6)

                return unittest.defaultTestLoader.loadTestsFromTestCase(ExampleTests)

        class ExampleUser(ExampleExercise):
            def solve(self, value: int) -> int:
                return value * self.factor

        class ExampleModel(ExampleExercise):
            def solve(self, value: int) -> int:
                return self.factor * value

        self.definition = ExampleExercise
        self.user_class = ExampleUser
        self.model_class = ExampleModel
        self.module = ModuleType("_example_user_solution")
        self.module.UserSolution = ExampleUser
        self.registration = RegisteredExercise(ExampleExercise, self.module.__name__)

    def run_main(self, args: list[str]) -> tuple[int, str]:
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            code = main(args)
        return code, output.getvalue()

    def test_definitions_and_missing_overrides_are_abstract(self):
        for definition in (Exercise, DNAExercise, MotifExercise, self.definition):
            with self.subTest(definition=definition.__name__),self.assertRaises(TypeError):
                definition()
        self.assertIsInstance(self.user_class(), self.definition)

    def test_user_and_reference_selection(self):
        with patch.dict(sys.modules, {self.module.__name__: self.module}):
            self.assertIsInstance(self.registration.load_solution("own"), self.user_class)
            self.assertIsInstance(self.registration.load_solution("model"), self.model_class)
        with self.assertRaisesRegex(ValueError, "Unknown solution"):
            self.registration.load_solution("invalid")

    def test_missing_and_unrelated_user_classes_are_rejected(self):
        with patch.dict(sys.modules, {self.module.__name__: self.module}):
            delattr(self.module, "UserSolution")
            with self.assertRaises(AttributeError):
                self.registration.load_solution("own")
            self.module.UserSolution = object
            with self.assertRaisesRegex(TypeError, "must subclass"):
                self.registration.load_solution("own")
            self.module.UserSolution = self.definition
            with self.assertRaises(TypeError):
                self.registration.load_solution("own")

    def test_wrong_reference_type_is_rejected(self):
        with patch.object(self.definition, "model_solution", return_value=object()),self.assertRaisesRegex(TypeError, "must be an instance"):
            self.registration.load_solution("model")

    def test_new_registration_runs_without_runner_changes(self):
        with patch.dict(EXERCISES, {"03": self.registration}), \
                patch.dict(sys.modules, {self.module.__name__: self.module}):
            code, output = self.run_main(["--exercise", "03", "--solution", "both"])
            self.assertEqual(code, 0)
            self.assertIn("Exercise 03 / own: PASS", output)
            self.assertIn("Exercise 03 / model: PASS", output)
            self.assertEqual(output.count("Ran 1 test"), 2)
            with patch(__name__ + ".run_for", return_value=True) as run:
                self.assertEqual(self.run_main(["--exercise", "all"])[0], 0)
                self.assertIn(unittest.mock.call("03", self.registration, "own"), run.call_args_list)
            self.assertIn("03: Example", self.run_main(["--list"])[1])

    def test_default_and_legacy_selection(self):
        for args, choices in (
            ([], ["own"]),
            (["--solution", "solve_this"], ["own"]),
            (["--solution", "model_solution"], ["model"]),
            (["--compare"], ["own", "model"]),
        ):
            with self.subTest(args=args), patch(__name__ + ".run_for", return_value=True) as run:
                self.assertEqual(self.run_main(args)[0], 0)
                self.assertEqual([call.args[0] for call in run.call_args_list], ["01"] * len(choices))
                self.assertEqual([call.args[2] for call in run.call_args_list], choices)

    def test_all_solution_modes(self):
        for selection, choices in (("own", ["own"]), ("model", ["model"]), ("both", ["own", "model"])):
            with self.subTest(selection=selection), patch(__name__ + ".run_for", return_value=True) as run:
                self.assertEqual(self.run_main(["--exercise", "all", "--solution", selection])[0], 0)
                self.assertEqual(
                    [(call.args[0], call.args[2]) for call in run.call_args_list],
                    [(exercise_id, choice) for exercise_id in EXERCISES for choice in choices],
                )

    def test_compare_continues_after_loading_failure(self):
        with patch.dict(EXERCISES, {"03": self.registration}):
            code, output = self.run_main(["--exercise", "03", "--compare"])
        self.assertEqual(code, 1)
        self.assertIn("Could not load own solution/tests", output)
        self.assertIn("ModuleNotFoundError", output)
        self.assertIn("Exercise 03 / own: FAIL", output)
        self.assertIn("Exercise 03 / model: PASS", output)

    def test_unimplemented_override_fails_without_reference_fallback(self):
        with patch.object(self.user_class, "solve", side_effect=NotImplementedError("Implement solve")), \
                patch.dict(sys.modules, {self.module.__name__: self.module}), \
                patch.dict(EXERCISES, {"03": self.registration}):
            code, output = self.run_main(["--exercise", "03", "--compare"])
        self.assertEqual(code, 1)
        self.assertIn("NotImplementedError: Implement solve", output)
        self.assertIn("Exercise 03 / own: FAIL", output)
        self.assertIn("Exercise 03 / model: PASS", output)

    def test_metadata_does_not_load_implementations(self):
        with patch.object(RegisteredExercise, "load_solution", side_effect=AssertionError("Loaded a solution")):
            self.assertEqual(self.run_main(["--list"])[0], 0)
            code, output = self.run_main(["--exercise", "all", "--describe"])
        self.assertEqual(code, 0)
        for exercise in EXERCISES.values():
            self.assertIn(exercise.definition.problem, output)

    def test_package_import_isolation(self):
        import subprocess
        from pathlib import Path

        script = """
import sys
import test

own_modules = {entry.own_module for entry in test.EXERCISES.values()}
model_modules = {
    entry.definition.__module__.rpartition(".")[0] + ".model_solution"
    for entry in test.EXERCISES.values()
}
solution_modules = own_modules | model_modules
assert not solution_modules.intersection(sys.modules)
assert test.main(["--list"]) == 0
assert test.main(["--exercise", "all", "--describe"]) == 0
assert not solution_modules.intersection(sys.modules)

selection = sys.argv[1]
if selection != "metadata":
    expected = own_modules if selection == "own" else model_modules
    for entry in test.EXERCISES.values():
        solution = entry.load_solution(selection)
        assert isinstance(solution, entry.definition)
    assert solution_modules.intersection(sys.modules) == expected

for old_module in ("dna_exercise", "motif_exercise", "solve_this",
                   "solve_this_02", "model_solution", "model_solution_02"):
    assert old_module not in sys.modules
"""
        for selection in ("metadata", "own", "model"):
            with self.subTest(selection=selection):
                result = subprocess.run(
                    [sys.executable, "-c", script, selection],
                    check=False, cwd=Path(__file__).resolve().parent,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_invalid_cli_arguments(self):
        for args in (
            ["--exercise", "missing"],
            ["--solution", "missing"],
            ["--compare", "--solution", "model"],
            ["--list", "--describe"],
        ):
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)

    def test_suites_are_fresh_and_empty_suites_fail(self):
        solution = self.user_class()
        first = solution.build_suite()
        second = solution.build_suite()
        self.assertIsNot(first, second)
        self.assertEqual(first.countTestCases(), 1)
        self.assertEqual(second.countTestCases(), 1)
        with patch.object(self.user_class, "build_suite", return_value=unittest.TestSuite()), \
                patch.dict(sys.modules, {self.module.__name__: self.module}), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertFalse(run_for("03", self.registration, "own"))


if __name__ == "__main__":
    sys.exit(main())
