# Streamlit Diagnostic Demo

A demo machine learning model diagnostic tool built with Streamlit that performs responsible AI checks to evaluate model quality, fairness, and interpretability.

## 🎯 Overview

This interactive demo provides automated diagnostics for machine learning models across four key dimensions:

- **📊 Calibration**: Assess model prediction confidence accuracy
- **⚖️ Fairness**: Detect potential bias and discrimination
- **🔍 Attribution**: Compare feature importance explanations
- **🎯 Simpler Modeling**: Evaluate if simpler models could achieve similar performance

## ⚠️ Disclaimer

This tool is intended for demo and educational purposes only. It should not be used for production model decisions. Always combine automated analysis with domain expertise and thorough testing.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (optional, for fairness checks)
- A binary classification dataset in CSV format (1=positive, 0=negative)
- A pre-trained scikit-learn compatible model (pickled)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Unlayer-AI/streamlit-diagnostic-demo.git
cd streamlit-diagnostic-demo
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables (optional):

```bash
export LLM_API_KEY="your_api_key"
export LLM_MODEL="your_model_name"
```

4. Run the application:

```bash
streamlit run diagnostic_demo/app.py
```

5. Open your browser at the indicated localhost URL (e.g. `http://localhost:8501`).

### Using Demo Data

The application includes sample data in the `demo_data/` folder:

- `adult.csv`: Adult income dataset
- `model.pkl`: Pre-trained model
- `train.csv` & `dev.csv`: Training and development sets

## 📋 Usage

### Basic Workflow

1. **Upload Data**: Provide a CSV file with your dataset
2. **Upload Model**: Provide a pickled scikit-learn compatible model
3. **Select Target Variable**: Choose the column your model predicts
4. **Configure LLM** (optional): Add OpenAI API key for fairness analysis
5. **Run Diagnostics**: Click "Diagnose!" to execute all checks`

## 📞 Contact

Want to learn more about responsible AI or integrate these diagnostics into your workflow?

**Unlayer AI** - Building transparent and trustworthy AI systems

- Learn about counterfactual explanations, integrated gradients, causality, and symbolic regression
- Explore enterprise solutions for responsible AI
- Get expert guidance on AI ethics and compliance
- Visit [https://unlayer.ai](https://unlayer.ai) for more information

## 📄 License

This project is licensed under MIT License with Attribution - see the [LICENSE](LICENSE) file for details.

---
