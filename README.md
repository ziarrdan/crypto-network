# crypto-network
In this project we create a network for long-lived cryptocurrencies from January 2020 to March 2022. Then we use well-known network science metrics and tools to study the evolution of this network, and the effect of external events on it. Additionally, we study the effects of bull-bear market cycles on the network and also investigate the evolution of the cross-correlations between different groups of cryptocurrencies.  

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
To run the code, simply run the main.py file under the main directory. The figures are generated under its corresponding  "pics" folder. Before running the main.py script for the first time, downloader.py could be used for getting cryptocurrencies price .csv files.
