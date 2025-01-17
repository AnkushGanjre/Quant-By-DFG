from taipy.gui.builder import Page, text

def create_page():
    with Page() as page:
        text("# Mean Reversion", mode="md")
    return page
