# Data Augmentation for Minority-Class Recognition

A reproducible image classification experiment evaluating whether conventional data augmentation improves recognition of an underrepresented class. The project compares imbalanced training, random oversampling, and offline augmentation on eight cat breeds from the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

The primary objective is to improve **Egyptian Mau recall** while preserving **macro-F1 across all eight classes**. A frozen ImageNet-pretrained ResNet-18 extracts image features; a multinomial logistic regression model performs classification.

## Experimental design

The selected dataset contains 910 original images: 120 images for each of Abyssinian, Bengal, Birman, Bombay, Ragdoll, Russian Blue, and Siamese, and 70 images for Egyptian Mau.

| Class group | Training images per class | Validation images per class | Test images per class |
| --- | ---: | ---: | ---: |
| Seven majority classes | 70 | 20 | 30 |
| Egyptian Mau | 20 | 20 | 30 |

Each iteration randomly assigns test images first, validation images second, and the remaining images to training. Three iterations use seeds `1729`, `2718`, and `31415`. Within each iteration, all conditions share the same original images and validation/test assignments.

| Condition | Minority-class training observations | Total training observations |
| --- | ---: | ---: |
| Imbalanced baseline | 20 originals | 510 |
| Oversampling | 20 originals + 50 repeated observations | 560 |
| Augmentation | 20 originals + 50 transformed images | 560 |

Oversampling and augmentation use the same source-image schedule. Augmentation combines random cropping, rotation, horizontal reflection, and brightness/contrast adjustments. Only training images are transformed. Validation and test sets remain balanced and unmodified.

ResNet-18 produces 512-dimensional features in evaluation mode. A `StandardScaler` is fitted to the original training split and shared across conditions. Logistic regression uses fixed settings: `C=1`, `solver="lbfgs"`, `tol=1e-7`, and `max_iter=3000`. Validation results are diagnostic; neither validation nor test results are used to change these settings.

## Requirements

- Python 3.12; a 64-bit interpreter is recommended.
- Windows, macOS, or Linux.
- CPU execution; no GPU is required.
- Internet access for dependency installation and, when rebuilding features or selecting data from scratch, model weights and dataset downloads.

`requirements.txt` contains the dependencies for cached-feature experiments, reporting, and result auditing. `requirements-features.txt` adds PyTorch and torchvision for feature extraction. It uses portable package specifications without a platform-specific wheel index or `+cpu` suffix. The exact environment used for the recorded experiment is preserved in `results/environment.json`; rebuilding features with a different environment can introduce numerical differences.

## Installation

From the project root, create a virtual environment:

```text
python -m venv .venv
```

If the interpreter is exposed as `python3` on your system, use `python3 -m venv .venv` instead. On Windows, `py -3.12 -m venv .venv` explicitly selects Python 3.12 when the Python launcher is installed.

Activate the environment using the command for your shell:

| Shell | Command |
| --- | --- |
| bash / zsh | `source .venv/bin/activate` |
| PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |

Install the standard dependencies:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activation is optional. The environment's interpreter can be used directly, for example `.venv/bin/python -m pip install -r requirements.txt` on macOS/Linux or `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` on Windows. This also avoids changing PowerShell execution policies. Install dependencies once before running the experiment; `main.py` does not install or update packages.

For feature extraction, install the additional dependencies:

```text
python -m pip install -r requirements-features.txt
```

## Data preparation

The original JPEG files belong in `data/selected/`, with filenames matching `data/selection.csv`. This source-and-results bundle includes the selection manifest, generated augmentations, and feature cache; the original JPEG subset is distributed separately. Place the existing image subset in this directory before running `experiment.py`.

Alternatively, reconstruct the subset and features from the official dataset using the additional feature-extraction dependencies:

```text
python experiment.py --prepare --archive ../images.tar.gz --recompute-features
```

If the archive is absent, the script downloads it from Oxford (approximately 755 MiB). The fixed selection seed is `20261018`. The command also rebuilds the splits and augmentations and executes all nine experiments. Pretrained ResNet-18 weights are downloaded automatically into `cache/torch/` when needed.

## Running the experiment

With original images, saved manifests, augmentations, and `features.npz` available, activate the virtual environment and run:

```text
python main.py
```

`main.py` runs `experiment.py`, `make_report.py`, and `audit_results.py` in order, using the same Python interpreter as the one running `main.py`. It stops and reports an error if any step fails. The default experiment validates the inputs, loads the cached features, and trains all nine classifiers. It does not require PyTorch. Reporting and auditing can also be run independently on the supplied results without the original JPEG files.

To recompute image features:

```text
python experiment.py --recompute-features --threads 4
```

To regenerate splits and augmentations and recompute features:

```text
python experiment.py --rebuild-data --recompute-features
```

Both commands require `requirements-features.txt`. Run `python main.py` afterwards to rerun the cached-feature experiment, update the report, and validate the results. Commands overwrite the corresponding generated outputs in `results/` and `report.md`; preserve a copy when comparing separate runs.

## Meeting 2 — Text-to-Image Generation

Meeting 2 is a separate workflow; `main.py` continues to run the completed Meeting 1 experiment. It follows the course template's 110-image pools, fixed generation seeds, 55-image synthetic budget, validation-first pool selection, and test-only-after-selection rule. It retains this repository's frozen ResNet-18 feature extractor and logistic regression rather than replacing the Meeting 1 classifier with the template's fine-tuned EfficientNet.

Install the additional dependencies once:

```text
python -m pip install -r requirements-meeting2.txt
```

Text-to-image generation is very slow on a CPU. Use a CUDA-capable GPU or a hosted GPU environment; Internet access and permission to download the Stable Diffusion 1.5 weights from Hugging Face are required. If the ResNet-18 weights are not already cached, training also downloads them. Generation refuses to run on CPU unless explicitly opted in with `--allow-cpu`. `--low-vram` enables memory-saving options on supported CUDA GPUs.

1. Generate the base pool and three controlled variations. Write the prompt(s) yourself; repeat `--prompt` to cycle through the same prompts in every pool. The base uses 30 steps and guidance 7.5; each other pool changes only guidance or steps.

   ```text
   python meeting2.py generate --prompt "Photorealistic photograph of a single Egyptian Mau cat with a short silver spotted coat and green eyes, natural proportions, soft daylight."
   ```

   This creates `m2-base`, `m2-guidance-2`, `m2-guidance-15`, and `m2-steps-10` under `pools/`, with 110 attempts per pool, a settings file, a JSONL manifest, and a contact sheet. The generator records errors and safety-blocked attempts instead of silently discarding them. Resume with the same command and settings; existing attempt records are kept.

2. Review each `pools/<name>/contact-sheet.jpg`. Note wrong breeds/coats, implausible anatomy, illustration-like images, repeated poses or backgrounds, and failed or safety-blocked attempts. Do not remove low-quality images or their manifest records.

3. Train the real-only baseline and all generated pools, then inspect the validation report and paired comparison plot:

   ```text
   python meeting2.py train
   python meeting2.py report
   ```

   Each pool adds 55 randomly selected, usable images per split. Five paired holdout seeds are used; the first three reuse the Meeting 1 seeds and the other two extend the comparison. Since logistic regression is deterministic, these are five data splits rather than five random model initializations. The existing data has 20 Egyptian Mau training images, so adding 55 produces 75 minority observations; this differs from the template's 15+55 balance. The baseline is refit on the same five splits for paired comparisons.

4. Choose one pool using **validation results and contact-sheet review**, and provide your justification when evaluating its test set. For example:

   ```text
   python meeting2.py test --pool m2-guidance-2 --reason "Highest validation Mau recall without reducing macro-F1; contact sheet shows mostly plausible spotted cats."
   ```

   Do not use test results to choose or change the pool. This command evaluates only the selected pool and its matched real-only baseline, then updates `results/meeting2/report.md` and the test comparison plot.

Generated pools, manifests, feature caches, and results are local experiment outputs and are ignored by Git. Keep them when sharing the completed experiment. `results/meeting2/report.md` contains validation tables, paired changes, contact-sheet links, and (after selection) test results; `pools/<name>/manifest.jsonl` records every generation attempt and its parameters.

## Recorded results

Mean test performance across the three recorded iterations:

| Condition | Egyptian Mau recall | Macro-F1 |
| --- | ---: | ---: |
| Imbalanced baseline | 67.78% | 84.25% |
| Oversampling | 72.22% | 84.83% |
| Augmentation | 78.89% | 85.95% |

Augmentation improved minority-class recall relative to the baseline in all three iterations. The mean improvement was 11.11 percentage points, accompanied by a 1.70-point increase in macro-F1. Relative to oversampling, augmentation improved mean recall by 6.67 points.

These results apply to the chosen data subset, transformations, and model. Each test split contains only 30 minority-class images, and test sets may overlap across iterations. Three repeated holdouts do not support strong claims of statistical significance or guarantee improvements in other settings.

## Project structure

| Path | Purpose |
| --- | --- |
| `main.py` | Run the experiment, generate the report, and audit results in sequence |
| `meeting2.py` | Generate Meeting 2 pools, train validation comparisons, and test a selected pool |
| `experiment.py` | Data selection, splits, augmentation, feature extraction, training, and evaluation |
| `make_report.py` | Report, summary statistics, and comparison figure generation |
| `audit_results.py` | Independent checks of saved predictions, metrics, and training observations |
| `requirements.txt` | Dependencies for cached-feature experiments and analysis |
| `requirements-features.txt` | Additional feature-extraction dependencies |
| `requirements-meeting2.txt` | Diffusion-generation and image-feature dependencies for Meeting 2 |
| `data/selection.csv` | Original-image selection, labels, and data fingerprints |
| `data/selected/` | Original JPEG subset, supplied separately or reconstructed from Oxford |
| `data/augmented/` | 50 generated training images per iteration |
| `manifests/` | Split assignments, augmentation sources, and transformation parameters |
| `features.npz` | Cached features, input paths, and input fingerprints |
| `results/metrics.csv` | Validation and test metrics for every iteration and condition |
| `results/iteration_*/` | Predictions, confusion matrices, classifier parameters, and training observations |
| `results/environment.json` | Provenance of the recorded feature-extraction environment |
| `protocol.md` | Detailed experimental protocol in Polish |
| `report.md` | Detailed results and interpretation in Polish |
| `DATA_LICENSE.md` | Dataset attribution and image licensing information |

## Reproducibility and validation

Automated checks verify class counts, split separation, original-image integrity, and training-only augmentation sources. Input fingerprints prevent the reuse of a feature cache after its images have changed. These checks are part of the experimental protocol; there is no separate distribution checksum file.

`audit_results.py` independently recomputes the primary metrics from saved predictions and checks that validation/test assignments agree across conditions. Stored classifier parameters use NumPy arrays rather than pickle. The code uses `pathlib` and UTF-8 text files; manifest paths use forward slashes on every operating system.

Exact duplicate detection does not rule out near-duplicate photographs or repeated animal identities. Random seeds control the data splits and transformations, but results may still vary slightly across numerical library versions and platforms.

## References and data attribution

- [Dataset Protocol and Real-Data Baselines — course assignment](https://patrykbart.faculty.wmi.amu.edu.pl/teaching/synthetic-data-generation-with-diffusion-models/meeting-1/)
- [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar. *Cats and Dogs*. IEEE CVPR, 2012.
- [torchvision ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [Course template repository](https://github.com/patrykbart/synthetic-data-course-template)
- [Meeting 2 assignment](https://patrykbart.faculty.wmi.amu.edu.pl/teaching/synthetic-data-generation-with-diffusion-models/meeting-2/)

Image attribution and the dataset's CC BY-SA 4.0 terms are described in `DATA_LICENSE.md`. These data terms do not establish a separate license for the project source code.
