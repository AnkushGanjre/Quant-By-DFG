from taipy.gui.builder import Page, text

def create_page():
    with Page() as page:
        text("# Simple Moving Average", mode="md")
    return page
