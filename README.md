# AI-Driven Procurement Intelligence System

A Streamlit-based web application that uses machine learning to forecast product
demand and recommend the most profitable procurement strategy.

## 🚀 Features
- Sales data analysis and visualization
- AI-based demand forecasting
- Safety stock calculation
- Supplier profitability comparison
- Automated PDF report generation

## 🛠️ Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- ReportLab

## 📂 Input Data Format

### Sales Data CSV
Required columns:
- date
- shop_name
- product_name
- units_sold

### Supplier Data CSV
Required columns:
- product_name
- supplier_name
- cost_per_unit
- max_capacity

## ▶️ How to Run Locally

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run api.py