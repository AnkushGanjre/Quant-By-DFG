from taipy.gui import Gui

# Define the content for the pages
main_page = """
# Welcome to Quant By DFG

This is a simple web app built using **Taipy**. Future development will include features like:
- Stock price prediction.
- Strategy backtesting.
- Comprehensive dashboards for quant finance analysis.

Visit the **Dashboard** for an overview (placeholder for now).

[Go to Dashboard](dashboard)
"""

dashboard_page = """
# Dashboard

Welcome to the dashboard page. Future updates will display:
- Stock data visualizations.
- Backtesting results.
- Predictive analytics.

Stay tuned for more!
"""

# Initialize the Taipy GUI
gui = Gui()

# Add pages to the GUI
gui.add_page("index", main_page)
gui.add_page("dashboard", dashboard_page)

# Run the application
if __name__ == "__main__":
    gui.run(host="0.0.0.0", port=10000)  # Use Render's default port