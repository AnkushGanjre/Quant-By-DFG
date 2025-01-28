import taipy.gui.builder as tgb


with tgb.Page() as home_page:
    tgb.text("# **Quant By** DFG", mode="md")
    tgb.text(
        "Home Page."
    )
    tgb.html("br")
