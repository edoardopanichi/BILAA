# Evolutionary Algorithm to play chess

***Abstract:*** The aim was to develop an evolutionary algorithm able to learn the game of chess starting from zero background. The algorithm relies on four crucial components: an evaluation function, the Monte Carlo search tree, a fitness function, and the evolutionary algorithm itself that determines how the agents have evolved generation after generation. With the code ready, I trained ten agents making them evolve for four generations, keeping the population size fixed at twenty specimens. Then these ten agents have been employed as the starting point for new training of fifteen generations with a population size equal to ten. The evaluation of the improvements has been conducted using two metrics: wins per generation and moves against Stockfish.

If you want more information read the paper related at https://www.overleaf.com/read/cmwhybhqdrfr.



> **! Disclaimer !**
Before running the code follow the step illustrate in the section of this README called "Install Stockfish".

## How to use the code
1. The necessary packages are listed in the file [requirements.txt](requirements.txt). If anything else is needed, the compiler will tell you more about it.

2. Follow the instructions in the section of this README called "Install Stockfish".

3. Open the [main.ipynb](./src/main.ipynb) and follow the markdown description to understand what each section does.

## Structure of the code
The code is subdivided into 9 files, I will quickly mention what each file does. Each file should be self-explanatory thanks to the comment inside.

1. [chess_functions.py](./src/chess_functions.py): inside it, there are different functions and classes related to chess. For example, the Stockfish engine and its functionality are here defined.

2. [evaluation_class.py](./src/evaluation_class.py): The class here contained defines the neural network used by the evaluation function. It also contains the functions that allow the NN to interpret the board position.

3. [evolutionary_algorithm.py](./src/evolutionary_algorithm.py): This file contains the EA itself, thus here is decided how the new generation are created.

3. [fitness_function.py](./src/fitness_function.py): In this file, it is contained the fitness function used to determine the fitness score of the agents in each generation. Only the fittest agents will survive in the next generation.
In particular, this fitness function works by simulating multiple matches against the agents and increasing the fitness scores of the winners while decreasing them for the losers.

4. [main.ipynb](./src/main.ipynb): This is the core of the project. Progressive implementation of the algorithm is shown cell after cell to show the differences. In the main, all the other files are used together.

5. [monte_carlo_search_tree.py](./src/monte_carlo_search_tree.py): A chess algorithm requires an algorithm to explore the possible moves. In my project, this is done by a Monte Carlo search tree. In particular, inside this file, two versions of it are implemented: the complete one, and a simplified version. In the training of the EA, the simplified version is used to save time.

6. [plot_wins.py](./src/plot_wins.py) and [plot_moves.py](./src/plot_moves.py): To generate some evaluation metrics it is possible to use these two files that create simple graphs starting from a given set of data.

7. [store_load_models.py](./src/store_load_models.py): A simple function here allows us to save the results of a run and re-used in a second moment.




## Install Stockfish:

> Tested on Ubuntu 22.04 and macOS 12.4.

- download the repository: https://github.com/official-stockfish/Stockfish
- Execute the following commands in the terminal:
    ```bash 
        cd src
        make help
        make net
        make build ARCH=x86-64-modern
    ```
- Then if you move the folder Stockfish-master into the folder of this repository, the code will work correctly. Otherwise, you need to modify the absolute path where the stockfish engine is initialized. For example: Stockfish(path="/Users/edoardo/Downloads/Stockfish-master/src/stockfish")

## Documentation: 
Some libraries used in this project:
- Stockfish library: https://pypi.org/project/stockfish/
- Chess on python: https://python-chess.readthedocs.io/en/latest/

## Sources
Some useful sources to better understand the project:
- https://towardsdatascience.com/building-a-chess-ai-that-learns-from-experience-5cff953b6784
- https://arxiv.org/pdf/1711.08337.pdf
- https://medium.com/@ishaan.gupta0401/monte-carlo-tree-search-application-on-chess-5573fc0efb75 
- Random ELO: https://chess.stackexchange.com/questions/6508/what-would-be-the-elo-of-a-computer-program-that-plays-at-random
