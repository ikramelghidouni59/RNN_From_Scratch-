# RNN and LSTM from Scratch (NumPy)

Practical work (TP) – Master IAA

Implementation of the forward and backward passes (BPTT) of a basic RNN and an LSTM in NumPy.

## Contents
- `RNN_From_Scratch.ipynb` – notebook with the 8 implemented functions
- `rnn_utils.py` – `sigmoid` and `softmax`
- `public_tests.py` – unit tests for the forward routines
- `report/` – IMRaD report (LaTeX source and PDF)

## Run
pip install numpy jupyter
jupyter notebook RNN_From_Scratch.ipynb
Run all cells from the top (keep the .py files in the same folder as the notebook).

## Results
- 4/4 forward unit tests pass.
- The 4 backward routines match the reference gradients (8–12 significant digits).

## Author
EL GHIDOUNI IKRAM
