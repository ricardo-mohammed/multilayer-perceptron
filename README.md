# Multilayer Perceptrons — Group Assignment

**Group:** Senay Teweldebrhan, Ricardo, Juan, and Zeyneb.

This project addresses both challenges in the supplied `MultiLayerPerceptrons_Workshop.ipynb`, using `MultiLayeredPerceptrons.ipynb` as supporting course material.

## Files

- `MLP_Group_Assignment.ipynb`: explanations, implementations, experiments, plots, answers, and reflection.
- `requirements.txt`: Python dependencies (not a fully pinned environment).
- `results/`: measured NumPy experiment results, reflection, and exported figures. The notebook generates Challenge 1 CSVs and plots when that section is run.

## Execution status — read before submission

**Challenge 2 has been executed**, including the finite-difference gradient check, eight experiment configurations, loss curves, decision surface, activation heatmaps, gradient diagnostics, and reflection. Its outputs are embedded in the notebook.

**Challenge 1 is implemented and syntax-checked but has not been executed here.** TensorFlow and PyTorch were unavailable; an attempted dependency installation could not resolve TensorFlow. Framework accuracy and speed are intentionally not invented. Before submission, run the full notebook in a configured environment, review the generated framework comparison and experiment tables, and save the notebook with its outputs.

The project is a complete implementation draft with verified NumPy results, not a claim that every framework has been runtime-tested. The actual Challenge 1 results remain a required execution step.

## Running in a notebook environment

1. Extract the ZIP first and keep the folder contents together.
2. Open `MLP_Group_Assignment.ipynb` in Jupyter, VS Code, or Google Colab.
3. Install missing dependencies using the commented `%pip` line in the first code cell. Restart the kernel after installation.
4. Use **Restart Kernel and Run All**. CPU execution is selected deliberately for the framework comparison.
5. MNIST downloads automatically on first run; internet access is needed for that download and package installation.
6. Allow all 12 framework runs to finish: three baseline runs plus three changed configurations for each framework. Runtime depends on the machine.
7. Review results and save the notebook. In Colab, download the executed notebook and the generated `results` files before the session ends.

For a local environment, use a Python version supported by both TensorFlow and PyTorch. Python 3.11 or 3.12 is a practical starting point; package/platform support should be checked if installation fails.

```bash
python -m venv .venv
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m jupyter lab
```

macOS/Linux activation:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

If the thread-setting cell reports that TensorFlow is already initialized, restart the kernel and run from the top. CPU wheel availability can vary by platform; use the official TensorFlow installation instructions if `tensorflow-cpu` is unavailable for your machine. Do not repeatedly run that setup cell after TensorFlow has already performed operations.

To study only Challenge 2 without TensorFlow/PyTorch, run the shared setup cell and then the Challenge 2 cells. NumPy, pandas, Matplotlib, scikit-learn, and IPython are sufficient for that section.

## Challenge 1: controlled comparison

The notebook compares Keras's `fit` interface, PyTorch's explicit loop, and TensorFlow's custom `GradientTape` loop. Keras uses a TensorFlow backend; these are three implementation approaches, not three entirely independent computational systems.

| Setting | Baseline |
|---|---|
| Dataset | MNIST digits 0–9 |
| Split | 54,000 train; 6,000 validation; 10,000 test |
| Features | 784 flattened pixels, scaled to [0, 1] |
| Hidden layers | 128 ReLU, 64 ReLU |
| Output | 10 classes; probabilities or loss-compatible logits |
| Optimizer | Adam, learning rate 0.001, epsilon 1e-7 |
| Batch size / epochs | 64 / 5 |
| Initialization / ordering | Shared initial weights and shared epoch permutations |
| Timing | Training and per-epoch validation; includes tracing; excludes download and final test |
| Hardware | CPU, two requested compute threads |

The separate experiments change batch size to 128, learning rate to 0.01, and epochs to 10. Only validation data is used to compare these variants. The predefined baseline test evaluations support the workshop's requested framework table; no test-driven tuning is performed.

The notebook answers `val_accuracy`, `val_loss`, `images`, `labels`, `criterion`, `optimizer`, `@tf.function`, `loss_object`, and `optimizer_tf`. It generates the required speed/accuracy/pros/cons table and measured talking points. Keras is proposed as the preferred interface for this standard task on usability grounds, not because it is guaranteed fastest or most accurate.

## Challenge 2: from-scratch implementation

- No framework layers or automatic differentiation are used in the NumPy implementation.
- Dataset: 1,000 noisy two-moons samples; stratified 600/200/200 train/validation/test split.
- Scaling is fitted only on training features.
- Baseline: 2 inputs → 16 ReLU → 8 ReLU → 1 sigmoid; full-batch gradient descent for 1,500 updates.
- Forward and backward equations use rows as samples, equivalent to the worksheet's transposed convention.
- Stable binary cross-entropy is evaluated from logits with `logaddexp`.
- All weights and biases in a small network are checked against central finite differences.
- Eight configurations compare activation, learning rate, depth, and initialization.
- The model with lowest final validation loss receives one held-out test evaluation.
- Figures show training losses, validation loss, the decision surface, forward activations, backward gradients, and initial gradient norms.
- A 150–200-word reflection is generated from the actual experiment results and saved as `results/reflection.md`.

The optional NumPy-versus-framework extension is not included. It is not a required deliverable in the workshop.

## Proposed group review allocation

This is a suggested division for learning and review, not a record of individual contributions already performed.

| Member | Suggested focus |
|---|---|
| Senay Teweldebrhan | Data preparation, Keras, comparison interpretation |
| Ricardo | PyTorch training loop and fair comparison controls |
| Juan | TensorFlow custom loop, timing, and result tables |
| Zeyneb | NumPy backpropagation, visualizations, and reflection |

Everyone should explain the complete prediction → loss → gradient → update cycle. Replace the proposed allocation with actual contributions if your instructor requests a contribution statement.

## Final review checklist

- [ ] Run Challenge 1 and all hyperparameter experiments successfully.
- [ ] Save the notebook with current outputs and inspect the generated tables.
- [ ] Discuss observed speed and accuracy without treating one run as a universal ranking.
- [ ] Review the NumPy equations, gradient check, flow plots, and reflection together.
- [ ] Confirm member names, course details, and instructor-specific submission requirements.
- [ ] Record actual contributions and disclose assistance if required by course policy.
- [ ] Submit the notebook and any other files explicitly requested by the instructor.

## Sources and interpretation

Primary requirements come from the two supplied course notebooks. Official references used to check implementation semantics:

- [Keras model training APIs](https://keras.io/api/models/model_training_apis/)
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [TensorFlow tf.function](https://www.tensorflow.org/guide/function)

The original workshop's example comparison was not copied as measured evidence. Its inconsistent preprocessing and training-set sizes are corrected here. Framework speed depends on hardware, runtime versions, tracing, threading, batching, and implementation choices. Multiple seeds and repeated timings would be needed for stronger conclusions.
