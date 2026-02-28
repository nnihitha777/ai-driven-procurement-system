
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# PDF Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import datetime

st.set_page_config(page_title="AI-Driven Procurement Engine", layout="wide")


st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
AI-Driven Procurement Intelligence System
</h1>
""", unsafe_allow_html=True)

st.markdown("---")


st.sidebar.header("⚙ Configuration Panel")

st.sidebar.markdown("### 📂 Upload Your Data Files")

st.sidebar.info("""
Sales CSV must contain columns:
- date
- shop_name
- product_name
- units_sold

Supplier CSV must contain columns:
- product_name
- supplier_name
- cost_per_unit
- max_capacity
""")

sales_file = st.sidebar.file_uploader(
    "Upload Sales Data CSV",
    type=["csv"]
)

supplier_file = st.sidebar.file_uploader(
    "Upload Supplier Data CSV",
    type=["csv"]
)

# =====================================================
# HELPER FUNCTION TO CLEAN COLUMNS
# =====================================================

def clean_columns(df):
    df.columns = [col.strip().lower() for col in df.columns]
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.lower()
    return df

# =====================================================
# LOAD SALES DATA
# =====================================================

if sales_file is not None:
    data = pd.read_csv(sales_file)
    data = clean_columns(data)

    required_sales_cols = {"date", "shop_name", "product_name", "units_sold"}

    if not required_sales_cols.issubset(set(data.columns)):
        st.error("❌ Sales CSV missing required columns.")
        st.stop()

    if data.empty:
        st.error("❌ Sales CSV is empty.")
        st.stop()

    data['date'] = pd.to_datetime(data['date']).dt.date

else:
    st.warning("Please upload Sales Data CSV to continue.")
    st.stop()

# =====================================================
# LOAD SUPPLIER DATA
# =====================================================

if supplier_file is not None:
    suppliers = pd.read_csv(supplier_file)
    suppliers = clean_columns(suppliers)

    required_supplier_cols = {
        "product_name",
        "supplier_name",
        "cost_per_unit",
        "max_capacity"
    }

    if not required_supplier_cols.issubset(set(suppliers.columns)):
        st.error("❌ Supplier CSV missing required columns.")
        st.stop()

    if suppliers.empty:
        st.error("❌ Supplier CSV is empty.")
        st.stop()

else:
    st.warning("Please upload Supplier Data CSV to continue.")
    st.stop()

# =====================================================
# SIDEBAR SELECTION
# =====================================================

shop_list = sorted(data['shop_name'].unique())
product_list = sorted(data['product_name'].unique())

selected_shop = st.sidebar.selectbox(
    "Select Shop",
    ["-- Select Shop --"] + shop_list
)

selected_product = st.sidebar.selectbox(
    "Select Product",
    ["-- Select Product --"] + product_list
)

selling_price = st.sidebar.number_input(
    "Enter Selling Price per Unit (₹)",
    min_value=0.01,  # Enforce positive selling price
    step=1.0,
    format="%.2f"
)

st.markdown("---")

# =====================================================
# MAIN LOGIC
# =====================================================

if (
    selected_shop != "-- Select Shop --"
    and selected_product != "-- Select Product --"
):

    filtered = data[
        (data['shop_name'] == selected_shop) &
        (data['product_name'] == selected_product)
    ].copy()

    if filtered.empty:
        st.warning("⚠ No sales data available for this selection.")
        st.stop()

    filtered = filtered.sort_values("date")

    st.subheader("📊 Recent Sales Overview")
    st.dataframe(filtered.tail(10))
    st.line_chart(filtered.set_index('date')['units_sold'])

    # =====================================================
    # AI FORECAST
    # =====================================================

    filtered = filtered.reset_index(drop=True)
    filtered['day_index'] = np.arange(len(filtered))

    X = filtered[['day_index']]
    y = filtered['units_sold']

    # Cache model to avoid retraining on widget changes
    @st.cache_data
    def train_model(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model

    model = train_model(X, y)
    r2_score = model.score(X, y)

    next_day = pd.DataFrame([[len(filtered)]], columns=['day_index'])
    predicted_demand = max(0, int(model.predict(next_day)[0]))  # avoid negative

    future_days = 7
    future_indices = np.arange(len(filtered), len(filtered) + future_days).reshape(-1, 1)
    future_predictions = model.predict(future_indices)
    future_predictions = np.maximum(0, future_predictions).astype(int)  # avoid negative

    safety_stock = int(predicted_demand * 0.2)
    final_order_quantity = predicted_demand + safety_stock

    # =====================================================
    # TREND VISUALIZATION
    # =====================================================

    fig, ax = plt.subplots()
    ax.plot(filtered['day_index'], filtered['units_sold'], label="Actual Sales")
    ax.plot(filtered['day_index'], model.predict(X), label="AI Trend Line")
    ax.plot(future_indices, future_predictions, linestyle="--", label="7-Day Forecast")
    ax.set_xlabel("Day Index")
    ax.set_ylabel("Units Sold")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # =====================================================
    # KPI DISPLAY
    # =====================================================

    st.subheader("📈 AI Forecast Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Demand (Next Day)", predicted_demand)
    col2.metric("Safety Stock (20%)", safety_stock)
    col3.metric("Final Order Quantity", final_order_quantity)
    st.metric("Model Accuracy (R² Score)", f"{r2_score:.3f}")
    std_dev = int(filtered['units_sold'].std())
    st.metric("Demand Volatility (Std Dev)", f"{std_dev} units")

    st.subheader("📅 7-Day Demand Forecast")
    forecast_df = pd.DataFrame({
        "Day": range(1, future_days + 1),
        "Predicted Units": future_predictions
    })
    st.dataframe(forecast_df)

    required_qty = final_order_quantity
    st.markdown("---")

    # =====================================================
    # SUPPLIER ANALYSIS
    # =====================================================

    available_suppliers = suppliers[
        suppliers['product_name'] == selected_product
    ].copy()

    if available_suppliers.empty:
        st.error("❌ No suppliers available for this product.")
        st.stop()

    st.subheader("💰 Supplier Profit Analysis")
    strategy_results = []

    for _, row in available_suppliers.iterrows():
        supplier_name = row['supplier_name']
        cost = row['cost_per_unit']
        capacity = row['max_capacity']

        if capacity >= required_qty:
            total_cost = required_qty * cost
            total_revenue = required_qty * selling_price
            profit = total_revenue - total_cost
            strategy_results.append({
                "Strategy": f"Full order from {supplier_name}",
                "Profit": profit,
                "Cost": cost
            })

    ranking_df = pd.DataFrame(strategy_results)
    if ranking_df.empty:
        st.error("No supplier can fulfill the required quantity.")
        st.stop()
    ranking_df = ranking_df.sort_values(by="Profit", ascending=False)

    st.subheader("📊 Supplier Ranking by Profitability")
    st.dataframe(ranking_df)

    if strategy_results:
        best_strategy = max(strategy_results, key=lambda x: x['Profit'])
        st.info(f"💡 System Recommended Strategy: {best_strategy['Strategy']}")

        chosen_strategy = best_strategy['Strategy']
        selected_profit = best_strategy['Profit']
        selected_cost = best_strategy['Cost']

        total_revenue = required_qty * selling_price
        margin_percent = (selected_profit / total_revenue) * 100

        st.success(f"Final PROFIT: ₹ {selected_profit:.2f}")
        st.write(f"Profit Margin: {margin_percent:.2f}%")
        st.write(f"Minimum Selling Price for Break-Even: ₹ {selected_cost:.2f}")

        # =====================================================
        # PDF REPORT EXPORT
        # =====================================================

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("<b>AI-Driven Procurement Decision Report</b>", styles['Title']))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Shop: {selected_shop}", styles['Normal']))
        elements.append(Paragraph(f"Product: {selected_product}", styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph("Forecast Summary", styles['Heading2']))
        elements.append(Paragraph(f"Predicted Demand: {predicted_demand}", styles['Normal']))
        elements.append(Paragraph(f"Safety Stock: {safety_stock}", styles['Normal']))
        elements.append(Paragraph(f"Final Order Quantity: {required_qty}", styles['Normal']))
        elements.append(Paragraph(f"Model R² Score: {r2_score:.3f}", styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))

        # Add forecast table to PDF
        forecast_table_data = [["Day", "Predicted Units"]]
        for i, val in enumerate(future_predictions, 1):
            forecast_table_data.append([str(i), str(val)])
        table = Table(forecast_table_data, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph("Selected Strategy", styles['Heading2']))
        elements.append(Paragraph(f"{chosen_strategy}", styles['Normal']))
        elements.append(Paragraph(f"Projected Profit: ₹ {selected_profit:.2f}", styles['Normal']))
        elements.append(Paragraph(f"Profit Margin: {margin_percent:.2f}%", styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph(
            f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))

        doc.build(elements)
        buffer.seek(0)

        st.download_button(
            label="📄 Download Professional PDF Report",
            data=buffer,
            file_name="procurement_decision_report.pdf",
            mime="application/pdf"
        )