# Demo Stock Visualization

## Usage
- [Usage](#usage)
- [Demo Stock Visualization](#what-is-quant-by-dfg)
- [Directory Structure](#directory-structure)

## What is Quant By DFG

[Quant By DFG](https://github.com/AnkushGanjre/quant-by-dfg) is a stock data dashboard with interactive visual elements to visualize historical stock data, and make prediction for the stock within 1 to 5 years.

### Demo Type
- **Level**: Basic
- **Topic**: Taipy-GUI
- **Components/Controls**: 
  - Taipy GUI: selector, chart, toggle, expandable, table, button, partial

## How to run

This demo works with a Python version superior to 3.8. Install the dependencies of the *Pipfile* and run the *main_markdown.py*.

## Introduction
Normally, if you want to check out a stock's historical performance like the opening/closing price and trading volume, it is often the case that this is done manually through a google search. Now, using taipy GUI and the yahoo finance library, we could get these informations in just a fraction of a second, visualize it using taipy's tools, and make a dashboard that anyone can use easily. 

The goal of this demo is to show how easy it is to build a data visualization dash with taipy. A fully interactive, highly customized web application that can be done in under 120 lines of python codes, is simply unheard of before, until taipy is introduced. 

Regarding the predicting algorithm, we used the prophet library from Meta.

Feel free to play with the application, enter a new ticker to the list, and make the most well-informed decision with Taipy! 


## Directory Structure


- `src/`: Contains the demo source code.
- `CONTRIBUTING.md`: Instructions to contribute to _demo-stock-visualization_.
- `Pipfile`: File used by the Pipenv virtual environment to manage project dependencies.
- `README.md`: Current file.

## Contributing

Want to help build _Demo Stock Visualization_? Check out our [`CONTRIBUTING.md`](CONTRIBUTING.md) file.

## Code of conduct

Want to be part of the _Demo Stock Visualization_ community? Check out our [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) file.
