# Workplace Commitment Classifier — DeBERTa V4

Classify an employee's workplace reply as **No**, **Latent**, **Qualified**, or **Explicit commitment**, using the earlier conversation as context.

The exported V4 model scored **92/96 (95.83%) accuracy** and **95.87% macro F1** on a small, frozen, balanced test of fictional workplace examples. These are project evaluation results, not established production accuracy. Ambiguous replies can be wrong with high confidence.

## What is included

- `test_model.py`: interactive and command-line offline CPU prediction.
- `install_model.py` and `model_integrity.json`: verified model installation.
- `requirements.txt`: portable inference dependencies.
- `data/v4/`: 528 training, 64 validation, and 96 test examples plus split audit.
- `results/v4/`: recorded metrics, predictions, errors, training logs, and plots.
- `notebooks/Latent_Commitment_DeBERTa_Training.ipynb`: historical notebook with code retained and outputs removed.
- `training/train_evaluate_v4.py`: final-stage V4 training/evaluation workflow, requiring the exact V3 checkpoint and a CUDA GPU.
- `docs/PROJECT_REFERENCE_V4.txt`: detailed history, settings, metrics, and limitations.

**The large model weights and Python environment are intentionally not included in Git.** Download the model ZIP separately; setup below is required after cloning. No retraining is needed to use the saved model.

## Setup for a teammate

Use **Python 3.12**, preferably 64-bit, and several GB of available memory/disk. The app uses CPU on macOS, Windows, and Linux. Only the original Mac deployment has been verified; other platforms require compatible PyTorch wheels and a local smoke test. Installation and the model download require internet; predictions run offline afterward.

### 1. Download or clone this repository

Use GitHub's **Code → Download ZIP**, extract it, and open a terminal inside the extracted project folder. Alternatively, clone your team's repository. The correct folder contains `test_model.py` and this README.

### 2. Create your own Python environment

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Windows, use `.\.venv\Scripts\python.exe` wherever later instructions say `python`; activation is optional. Do not copy someone else's `.venv` or `.python` directory.

`requirements.txt` pins Transformers and SentencePiece and allows compatible PyTorch versions. For the exact tested Mac package versions, see `docs/requirements-macos-tested.txt`. Do not install the Colab CUDA-specific Torch pin on a Mac.

### 3. Obtain the saved model

Download **[latent_commitment_workplace_v4_model.zip from the V4 release](https://github.com/NShenbagakrithika/latent-commitment-inference-model/releases/download/v4.0.0/latent_commitment_workplace_v4_model.zip)** (561,784,606 bytes, approximately 562 MB).

You can also find it on the [release page](https://github.com/NShenbagakrithika/latent-commitment-inference-model/releases/tag/v4.0.0). Download the named model ZIP asset, not GitHub's automatically generated source archive. Keep the downloaded ZIP outside the repository folder or leave it untracked.

Verify and install (replace the path with your downloaded ZIP):

```bash
python install_model.py "/path/to/latent_commitment_workplace_v4_model.zip"
```

Windows example:

```powershell
.\.venv\Scripts\python.exe install_model.py "$HOME\Downloads\latent_commitment_workplace_v4_model.zip"
```

The installer verifies the archive and all six model/tokenizer files. It installs only the files needed for inference, ignores pickled training state, and does not overwrite a different existing model. The installed folder is:

```text
best_model/
  config.json
  model.safetensors
  spm.model
  tokenizer_config.json
  special_tokens_map.json
  added_tokens.json
```

Expected model weights SHA256:

```text
fc3f77b652ee7231430eb95d90d300b6496983fc37a1c42670017d8936b2a633
```

### 4. Enter a conversation

```bash
python test_model.py
```

At **Context**, paste the preceding messages on **one line**, for example:

```text
[manager]: Can you send the revised client summary by three this afternoon?
```

At **Employee reply**, enter:

```text
Yes, I'll send the completed summary by three.
```

Expected label: `EXPLICIT_COMMITMENT`. The terminal displays all four class scores. Enter another example, or type `quit` at Context to exit. A multiline paste can be interpreted as separate prompt answers: use one line for context.

One-shot prediction:

```bash
python test_model.py --context "[manager]: Can you cover tomorrow's customer call?" --reply "No, I cannot take that call."
```

Add `--json` for structured output. The Python API is also available:

```python
from test_model import CommitmentModel
model = CommitmentModel()
print(model.predict("[manager]: Can you send the summary?", "Yes, I'll send it today."))
```

Inputs must include nonblank context and reply and fit within **256 model tokens**. Longer inputs are rejected, not silently truncated. Entered conversations are not saved by this application; terminal history/scrollback is managed by your terminal.

## Label definitions

| ID | Label | Project meaning |
|---|---|---|
| 0 | NO_COMMITMENT | No new undertaking; refusal, facts, past completed work, another person's promise, or merely social agreement |
| 1 | LATENT_COMMITMENT | Ownership implied through relevant preparation/action without direct acceptance |
| 2 | QUALIFIED_COMMITMENT | Actual undertaking limited by unresolved conditions, uncertainty, or partial scope |
| 3 | EXPLICIT_COMMITMENT | Direct unqualified acceptance/promise; already satisfied prerequisites do not make it conditional |

These are provisional definitions. Bare acknowledgements and availability statements may have multiple reasonable readings. Scores are uncalibrated softmax outputs, not guarantees of correctness.

## Verified training history

The current model did **not** inherit fine-tuned weights from the original separate 7,000-row experiment.

```text
Microsoft pretrained DeBERTa-v3-base
  → V2: 7,680 synthetic training rows; 3 epochs run, epoch 1 retained
  → V3: 232 workplace training rows; 6 epochs run, epoch 4 retained
  → V4: 528 workplace training rows; 6 epochs run, epoch 6 retained
```

V4's 528 rows contain all 368 V3 rows (232 training + 48 former validation + 88 former test) plus 160 new examples. V4 therefore uses new 64-row validation and 96-row test sets. There are 8,440 stage-level training-row uses and 8,208 distinct normalized training inputs across the three stages, excluding epoch repetition.

V4 settings: selected learning rate `5e-6`, AdamW, batch 8, accumulation 2, weight decay `.01`, warmup `.1`, FP16, max 256 tokens, seed `20261017`, Tesla T4. Cross-entropy plus `.2` times a context-pair softplus margin loss (margin 1). Whole-model fine-tuning. Checkpoint selection: `.8 * validation macro F1 + .2 * validation pair-both-correct`. Two fixed learning-rate candidates (`5e-6`, `1e-5`), with validation-only selection and lower-rate tie-break. See the reference and recorded logs for all stages.

## Recorded final evaluation

| Model | Same-test accuracy | Macro F1 |
|---|---:|---:|
| Current V4 DeBERTa | **95.83%** | **95.87%** |
| Previous V3 DeBERTa | 88.54% | 88.40% |
| TF-IDF full input | 90.63% | 90.53% |
| TF-IDF reply only | 75.00% | 73.74% |
| TF-IDF context only | 77.08% | 76.27% |

V4 macro precision: **96.18%**; macro recall: **95.83%**. Balanced support makes weighted metrics equal to macro metrics. Both members correct in **14/16 context pairs**. All 96 test examples were retained. Three of four errors had scores at least 90%.

Fresh validation/test exact normalized input, context, target, and group overlaps were checked; groups stayed within one split and metadata was excluded from inputs. These audits do not rule out semantic similarity, wording shortcuts, or author style. All later workplace examples are fictional and agent-authored, not independently annotated real workplace gold. The 95.83% result applies to this small test; later manual ambiguous examples revealed further errors.

## Training and historical notebook

Prediction does not require any training dependencies or a GPU.

For archival review, open the notebook in `notebooks/`. It contains multiple historical experiments; **do not use Run all just to test the model**. Outputs were removed for sharing; authoritative recorded results are in `results/v4/`. Historical cells have `/content` Colab paths, execution-order dependencies, and earlier dataset/checkpoint requirements. The original 10,000-row user ZIP is not bundled.

For the final V4 training stage only, use a CUDA GPU, the exact V3 contextual checkpoint, and the included frozen V4 splits:

```bash
cd training
python -m pip install -r requirements.txt
cd ..
python training/train_evaluate_v4.py --init-model /path/to/v3/contextual_model --output-dir runs/v4_reproduction
```

The required V3 weights hash is `a6f8b3fa7565136761183cacbd4d61e9b55e3eb411004368cdaf5130335d5009`. **That V3 checkpoint is not included in this inference package**; obtain the retained artifact if available or recreate it through the historical workflow. The V4 model ZIP cannot substitute for V3 when reproducing V4's original training. A new output directory is required. GPU/package nondeterminism can change rerun results. Training code is preserved and syntax-checked; packaging did not rerun training or claim new metrics. This dataset's test is now exposed; use a genuinely new held-out test when developing further improvements.

The proposed V5 ambiguous-reply workflow is not included as a trained model. It has no measured results and must not be confused with V4.

## Sharing with a teammate

Send your teammate [this repository](https://github.com/NShenbagakrithika/latent-commitment-inference-model) and ask them to follow the setup instructions above. The model is a separate [V4 release download](https://github.com/NShenbagakrithika/latent-commitment-inference-model/releases/tag/v4.0.0).

Keep `.gitignore`: model weights, archives, environments, caches, generated training runs, and common credential files should stay out of source commits. For future improvements, version new model artifacts separately and report their own validation/test results.

Base model: [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base). See `NOTICE.txt` for attribution and licensing status.

## Troubleshooting

- **Missing model file:** download the complete original model ZIP and run `install_model.py`.
- **Checksum mismatch:** the wrong model or an incomplete download was supplied; obtain the verified V4 artifact.
- **No matching package/wheel:** use Python 3.12 and the appropriate PyTorch installation for your operating system.
- **Long input:** shorten the preceding conversation to fit 256 model tokens.
- **Unexpected prediction:** check that context and target were entered separately. Do not treat a high score as proof; ambiguous language is a known weakness.
- **No Colab GPU / Drive authorization error:** local inference needs neither Colab nor Drive once the files have been downloaded.
