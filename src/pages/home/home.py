import taipy.gui.builder as tgb


with tgb.Page() as home_page:
    tgb.text("# **Dashboard**", mode="md")
    tgb.text(
        "Dashboard text here"
    )
    tgb.html("br")
    
