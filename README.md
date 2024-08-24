# crypto-network
In this project, network analysis has been employed to study the dynamic evolution of the cryptocurrency market from 1 January 2020 to 1 January 2024. This approach facilitates an in-depth exploration of the market’s response to several major events during this period, including the coronavirus disease of 2019 (COVID-19) pandemic and the bankruptcy of FTX, one of the largest cryptocurrency exchanges. The study focuses on analysing key network characteristics of the cryptocurrency market, namely: (a) degree centrality, (b) betweenness centrality, (c) clustering coefficient and (d) average path length. Additionally, we explore the co-movements within the market, categorising cryptocurrencies into functional groups for a comparative analysis. This approach enables us to examine shifts in the cryptocurrency network topology, providing insights into how different groups of cryptocurrencies interact with and influence each other. Through this network analysis, we aim to shed light on the intricate interrelationships among cryptocurrencies. The findings of this study are intended to provide investors with valuable insights, potentially guiding the development of more informed and strategic diversification strategies in the dynamic and evolving landscape of the cryptocurrency market. The findings of this project is published in the Journal of Interdisciplinary Economics in a paper titled [Dynamic Evolution Analysis of Cryptocurrency Market: A Network Science Study](https://journals.sagepub.com/doi/10.1177/02601079241265744).

## Required IDE and Libraries
The code in this repository was written and tested in PyCharm 2020.3 using the following libraries (the dependencies of each library is not mentioned).

Library | Version
--------------|------------
networkx* | 2.6.3
pycoingecko** | 2.2.0
matplotlib | 3.5.0
numpy | 1.20.3
pandas | 1.4.1
scipy | 1.7.3

\* https://networkx.org
\** https://github.com/man-c/pycoingecko

## How to Run
To run the code, simply run the main.py file under the main directory. The figures are generated under its corresponding  "pics" folder. Before running the main.py script for the first time, downloader.py could be used for getting cryptocurrency prices .csv files using CoinGeckoAPI API.

## Results
Sample Outputs:

|||
--------------|------------
|![Cryptocurrencies network topology](https://github.com/ziarrdan/crypto-network/blob/main/pics/composite0.png?raw=true) |
