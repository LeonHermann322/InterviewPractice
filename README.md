# PyTorch Live Coding Interview Practice

## Files
- `requirements.txt`: dependencies
- `solve_this.py`: candidate scaffold; implement the TODOs here
- `model_solution.py`: reference implementation
- `test.py`: tests selectable from the command line

## Setup
Use Python 3.10+ in a virtual environment if possible:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## Run the tests
Test your implementation:

```bash
python test.py --solution solve_this
```

Test the reference solution:

```bash
python test.py --solution model_solution
```

Run both suites:

```bash
python test.py --compare
```

The scaffold is expected to fail until you implement it. The reference solution should pass.
The tests cover encoding, input validation, model output shape, gradient flow, training,
and prediction behavior. Passing these tests is not a proof of biological usefulness or
generalization; the dataset is deliberately tiny and synthetic.

## Suggested interview rules
- Give yourself 20 minutes to debug/implement the functions.
- Avoid opening `model_solution.py` until you're finished.
- Use the error messages and failing tests as feedback, just as you would in a coding interview.
