from taipy.gui.builder import Page, text, toggle

def create_page():
    with Page() as page:
        text("# Quant By DFG", mode="md")
    return page
