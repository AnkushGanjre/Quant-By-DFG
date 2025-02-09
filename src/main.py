from taipy.gui import Gui
import taipy as tp

from pages.dashboard.dashboard import dashboard_page
from pages.value_at_risk.value_at_risk import valueAtRisk_page
from pages.option_pricing.option_pricing import optionPricing_page
from pages.root import root

# from config.config import Config


pages = {
    "/": root,
    "Dashboard": dashboard_page,
    "VaR": valueAtRisk_page,
    "OP": optionPricing_page,
}


if __name__ == "__main__":
    gui_multi_pages = Gui(pages=pages)

    tp.Orchestrator().run()

    gui_multi_pages.run(
        title="Quant By DGF", 
        use_reloader=True,      # Enable automatic reloading during development
        host="0.0.0.0",         # Set host to 0.0.0.0 (localhost)
        port=2562,
        margin="0px"            # Here all the CSS styling is set
        )
