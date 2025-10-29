# ThyroidClassification

Simple Flask service to predict thyroid disorder categories from clinical inputs.

## Quick start (Windows)
1. Create a virtual environment (if you haven't already):

```powershell
python -m venv venv
```

2. Activate the virtual environment:

```powershell
venv\\Scripts\\activate.bat
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Run the app:

```powershell
python app.py
```

The service will start on http://127.0.0.1:5000 by default.

## Quick start (macOS / Linux)
1. Create a virtual environment:

```bash
python3 -m venv venv
```

2. Activate the virtual environment:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python app.py
```

## Predict endpoint

POST http://127.0.0.1:5000/predict
- Content-Type: application/json
- Body: JSON object with patient features

Example request (exact command):

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"age\": 45, \"gender\": \"female\", \"T3\": 1.5, \"T4\": 110, \"TSH\": 2.5, \"on_thyroxine\": \"f\", \"query_on_thyroxine\": \"f\", \"on_antithyroid\": \"f\", \"sick\": \"f\", \"pregnant\": \"f\", \"thyroid_surgery\": \"f\", \"i131_treatment\": \"f\", \"query_hypothyroid\": \"f\", \"query_hyperthyroid\": \"f\", \"lithium\": \"f\", \"goitre\": \"f\", \"tumor\": \"f\", \"hypopituitary\": \"f\", \"psych\": \"f\"}"
```

Note: The boolean-like fields in the dataset/service use single-character values 't' (true) and 'f' (false) as strings.

## Input feature list
The model accepts the following fields in the JSON body:
- age (number)
- gender (string) — example: "male" or "female"
- T3 (number)
- T4 (number)
- TSH (number)
- on_thyroxine (string: "t" or "f")
- query_on_thyroxine (string: "t" or "f")
- on_antithyroid (string: "t" or "f")
- sick (string: "t" or "f")
- pregnant (string: "t" or "f")
- thyroid_surgery (string: "t" or "f")
- i131_treatment (string: "t" or "f")
- query_hypothyroid (string: "t" or "f")
- query_hyperthyroid (string: "t" or "f")
- lithium (string: "t" or "f")
- goitre (string: "t" or "f")
- tumor (string: "t" or "f")
- hypopituitary (string: "t" or "f")
- psych (string: "t" or "f")

Adjust or validate fields according to your specific model implementation in `app.py`.

## Response / Label mapping
The API returns a numeric prediction. The mapping is:
- 0: Normal
- 1: Hypothyroidism
- 2: Hyperthyroidism
- 3: Other disorder

A typical response might look like:
```json
{
  "prediction": 0,
  "label": "Normal"
}
```

(If your implementation returns only the integer, map it locally using the mapping above.)

## Troubleshooting
- If port 5000 is in use, either stop the conflicting service or modify `app.py` to run on a different port.
- Ensure your Python version matches the one used to create `requirements.txt` (commonly Python 3.8+).
- If dependencies are missing, run `pip install -r requirements.txt` inside the activated virtualenv.

## Development notes
- Ensure `app.py` loads the trained model and performs the same preprocessing steps used during training.
- If adding new features or changing preprocessing, update the README and any inference tests accordingly.

## License
Add a license file if you want to make usage terms explicit.