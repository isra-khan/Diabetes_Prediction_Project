# Diabetes Prediction using Machine Learning

An educational machine-learning project that uses a linear Support Vector
Machine (SVM) to classify diabetes outcomes from health measurements. The
repository includes both an interactive Streamlit web app and the original
Jupyter notebook.
<img width="1440" height="673" alt="image" src="https://github.com/user-attachments/assets/8d17e7ac-ff3c-4246-894c-dac05e3d914f" />

> **Medical disclaimer:** This project is for educational purposes only. Its
> predictions are not a diagnosis, medical advice, or a substitute for care
> from a qualified healthcare professional. The model's accuracy is not a
> measure of the probability that a particular prediction is correct.

## Features

- Enter eight measurements in a browser-based form.
- View the model's positive or negative class prediction.
- See the model's held-out test accuracy.
- Explore the training workflow in the Jupyter notebook.

The app's prediction is a class label, not a probability or clinical risk
assessment.

## Requirements

- Python 3.10 or later
- `pip`

## Run the web app

Open a terminal in the project directory and create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies and launch the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit (typically
<http://localhost:8501>) in your browser. Enter the measurements and click
**Predict**.

To stop the local server, press `Ctrl+C` in the terminal. The app is available
only while that server is running.

## Input measurements

Enter values in the units used by the dataset:

| Input | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| Blood pressure | Diastolic blood pressure |
| Skin thickness | Triceps skin fold thickness |
| Insulin | Two-hour serum insulin |
| BMI | Body mass index |
| Diabetes pedigree function | Diabetes pedigree function value |
| Age | Age in years |

## Model and evaluation

The app reads `diabetesdata.csv`, standardizes the input features, and trains a
linear SVM (`SVC`). It reports accuracy on a stratified 20% held-out split
using a fixed random seed. After evaluation, the prediction model is trained
on the full dataset. Accuracy is an aggregate measure on this dataset split
and does not guarantee a correct result for an individual.

## Run the notebook

Start Jupyter from the project directory so the notebook can find
`diabetesdata.csv`:

```bash
source .venv/bin/activate  # Activate the environment (Windows: .venv\Scripts\Activate.ps1)
jupyter lab
```

Then open `Diabetes Prediction using Support Vector Machines.ipynb` and run the
cells in order. The notebook and app both use the same CSV dataset.

## Project files

```text
app.py                                              Streamlit web app
diabetesdata.csv                                    Dataset
Diabetes Prediction using Support Vector Machines.ipynb
                                                   Model walkthrough
requirements.txt                                   Python dependencies
README.md                                           Project documentation
```
