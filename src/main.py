from taipy.gui import Gui
import taipy as tp

from pages.Home.home import home_page
from pages.VaR.VaR import VaR_page
from pages.root import root

# from config.config import Config


pages = {
    "/": root,
    "Home": home_page,
    "VaR": VaR_page,
}


if __name__ == "__main__":
    gui_multi_pages = Gui(pages=pages)

    tp.Orchestrator().run()

    gui_multi_pages.run(
        title="Quant By DGF", 
        use_reloader=True,      # Enable automatic reloading during development
        host="0.0.0.0",         # Set host to 0.0.0.0 (localhost)
        port=2562,
        margin="0px"
        )
