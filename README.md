# Stock Data Analysis

A comprehensive thesis project analyzing stock price data for major technology companies using Python and Jupyter Notebooks.

## Project Overview

This project contains detailed analysis and modeling of stock price data for five major tech companies:
- **AAPL** (Apple)
- **AMZN** (Amazon)
- **GOOGL** (Google/Alphabet)
- **MSFT** (Microsoft)
- **NVDA** (NVIDIA)

## Project Structure

```
├── code/                          # Jupyter notebooks with analysis
│   ├── ThesisCode_AAPL.ipynb
│   ├── ThesisCode_AMZN.ipynb
│   ├── ThesisCode_GOOGL.ipynb
│   ├── ThesisCode_MSFT.ipynb
│   └── ThesisCode_NVDA.ipynb
├── datasets/                      # Historical price data (CSV format)
│   ├── AAPL-Price History_20260331_0843.csv
│   ├── AMZN -Price History_20260331_0845.csv
│   ├── GOOGL-Price History_20260331_0842.csv
│   ├── MSFT -Price History_20260331_0844.csv
│   ├── NVDA-Daily-Price History_20260331_0839.csv
│   └── Price History_20260307_1934-clean.csv
└── README.md                      # This file
```

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required packages (see below)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/WilliamKesuma/Stock-Data-Analysis.git
cd Stock-Data-Analysis
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install jupyter pandas numpy matplotlib scikit-learn
```

### Running the Analysis

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to the `code/` directory and open any of the thesis notebooks (e.g., `ThesisCode_AAPL.ipynb`)

3. Run the cells to execute the analysis

## Data

All datasets contain historical price data for each stock, including:
- Opening price
- Closing price
- High/Low prices
- Volume
- Date information

The data files are located in the `datasets/` folder and are in CSV format for easy access and manipulation.

## Analysis

Each notebook contains comprehensive analysis including:
- Data exploration and visualization
- Statistical analysis
- Time series modeling
- Trend analysis
- Price predictions

## Author

William Kesuma

## License

This project is part of a thesis and may have specific academic usage restrictions. Please check with the author before using this work in any publication or commercial context.

## Contact

For questions or collaboration, please reach out via GitHub.
