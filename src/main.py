import taipy as tp
from taipy.gui import Gui, navigate
from pages.root import create_page as create_root_page
from pages.home.home import create_page as create_home_page
from pages.simpleMovingAverage.simpleMovingAverage import create_page as create_sma_page
from pages.meanReversion.meanReversion import create_page as create_mean_reversion_page

# Define themes
light_theme = {
    "palette": {
        "background": {
            "default": "#f0f4f8",  # Soft light gray background
        },
        "primary": {
            "main": "#4a90e2",  # Soft blue for primary elements
        },
        "text": {
            "primary": "#333333",  # Dark gray text for good readability
            "secondary": "#666666",  # Medium gray for less emphasized text
        }
    }
}

dark_theme = {
    "palette": {
        "background": {
            "default": "#1e1e1e",  # Dark gray background
        },
        "primary": {
            "main": "#4a90e2",  # Soft blue for primary elements
        },
        "text": {
            "primary": "#ffffff",  # White text for readability
            "secondary": "#b0b0b0",  # Light gray for less emphasized text
        }
    }
}

# Function to handle menu navigation
def menu_option_selected(state, action, info):
    page = info["args"][0]
    navigate(state, to=page)

# Initialize pages
pages = {
    "/": create_root_page(menu_option_selected),
    "home": create_home_page(),
    "simpleMovingAverage": create_sma_page(),
    "meanReversion": create_mean_reversion_page(),
}

app = Gui(pages=pages)

if __name__ == "__main__":
    # Start the Core application
    tp.Orchestrator().run()

    # Run the GUI application
    app.run(
        title="QuantByDFG",
        use_reloader=True,  # Enable automatic reloading during development
        host="0.0.0.0",     # Set host to 0.0.0.0
        port=5000,
        light_theme=light_theme,
        dark_theme=dark_theme,
    )