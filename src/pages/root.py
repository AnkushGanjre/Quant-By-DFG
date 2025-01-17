from taipy.gui import Markdown
from taipy.gui.builder import Page, menu, text, toggle

def create_page(menu_callback):
    with Page() as page:
        menu(
            label="Menu",
            lov=[
                ("home", "Home"),
                ("simpleMovingAverage", "Simple Moving Average"),
                ("meanReversion", "Mean Reversion"),
            ],
            on_action=menu_callback,
        )
        # text("# Quant By DFG\nWelcome to the app!", mode="md")
    return page

# root = Markdown('root.md')