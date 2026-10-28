"""Common contract for interview exercises; contains no reference solutions."""
from abc import ABC, abstractmethod
from typing import ClassVar
import unittest


class Exercise(ABC):
    """Subclass with problem text, tests, and exercise-specific abstract stubs.

    User and reference implementations subclass the exercise definition.
    Test suites exercise the current instance, regardless of its implementation.
    """

    title: ClassVar[str]
    problem: ClassVar[str]

    @classmethod
    @abstractmethod
    def model_solution(cls) -> "Exercise":
        """Create the reference implementation, importing it only on demand."""
        raise NotImplementedError

    @abstractmethod
    def build_suite(self) -> unittest.TestSuite:
        """Return a fresh suite testing this implementation."""
        raise NotImplementedError
