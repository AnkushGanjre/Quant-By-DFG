import taipy.gui.builder as tgb


with tgb.Page() as VaR_page:
    tgb.text("# **VaR &** CVaR", mode="md")
    tgb.text(
        "Value at Risk & Conditional Value at Risk"
    )
    tgb.html("br")