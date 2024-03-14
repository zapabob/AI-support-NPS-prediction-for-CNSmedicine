# Compound Activity Predictor

This project implements Graph Convolutional Networks (GCNs) and Random Forest models for predicting the activity (pIC50) of compounds from their SMILES or IUPAC names. Data is retrieved from the ChEMBL database. Additionally, we utilize quantum chemistry features, such as electron densities obtained from quantum chemistry calculations, to improve prediction accuracy.

## Asynchronous Processing

This project incorporates asynchronous processing to prevent the GUI from freezing during predictions. Users can continue interacting with the GUI while predictions are running in the background.

## Requirements

- Python 3.6 or higher
- pip

## Installation

1. Clone this repository or download it as a ZIP file.
git clone https://github.com/AI-support-NPS-prediction-for-CNS-medicine-with-AMD-GPU-programming-/compound-activity-predictor.git
2. Navigate to the project directory and install the required libraries.
cd compound-activity-predictor
pip install .
## Usage

1. Run the `compound-activity-predictor` command to launch the GUI application.
2. Enter a SMILES or IUPAC name in the input field and click the "Predict" button.
3. The 2D structure of the compound and the prediction results from the GCN and Random Forest models will be displayed.

### Asynchronous Execution

1. Clicking the "Predict" button will start the prediction process in a background thread.
2. The progress bar will display the progress of the prediction.
3. Once the prediction is complete, the results will be displayed in the text box and graph.

## Directory Structure
compound-activity-predictor/
├── data/
├── docs/
│   ├── motivation.md
│   ├── references.md
│   └── ...
├── models/
├── tests/
├── utils/
│   ├── init.py
│   ├── data_utils.py
│   ├── descriptor_utils.py
│   ├── model_utils.py
│   └── quantum_utils.py
├── main.py
├── gui.py
├── requirements.txt
├── README.md
└── setup.py
- `data/`: Directory for data files
- `docs/`: Directory for documentation files
- `models/`: Directory for saving trained models
- `tests/`: Directory for unit tests
- `utils/`: Directory containing utility modules
- `main.py`: Main script
- `gui.py`: GUI application code
- `requirements.txt`: List of required libraries
- `README.md`: Project overview and usage instructions
- `setup.py`: Project installation script

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).

## Contributing

Bug reports and pull requests are welcome. For more information, please see [CONTRIBUTING.md](CONTRIBUTING.md).
