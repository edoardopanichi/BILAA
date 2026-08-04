# Evolutionary Algorithm to play chess

***Abstract:*** The aim was to develop an evolutionary algorithm able to learn the game of chess starting from zero background. The algorithm relies on four crucial components: an evaluation function, a Monte Carlo search tree, a fitness function, and the evolutionary algorithm itself. The current training path uses balanced self-play, state-aware board inputs, terminal rewards, batched inference, and enhanced genetic variation. Evaluation should be performed after training against fixed tactical positions and Stockfish rather than benchmarking Stockfish after every generation.

If you want more information read the paper related at https://www.overleaf.com/read/cmwhybhqdrfr.



> **! Disclaimer !**
Before running the code follow the step illustrate in the section of this README called "Install Stockfish".

## How to use the code
1. Use Python 3.14 and create a virtual environment from the repository root:
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

   The project uses Keras 3 with the PyTorch backend because the stable TensorFlow package does not currently provide a native Windows Python 3.14 wheel.

2. Follow the instructions in the section of this README called "Install Stockfish".

3. Start Jupyter from the `src` directory so the notebook's module and model paths resolve correctly:
    ```powershell
    cd src
    New-Item -ItemType Directory -Force ..\.jupyter\config, ..\.jupyter\data, ..\.jupyter\runtime, ..\.jupyter\ipython, ..\.matplotlib | Out-Null
    $env:JUPYTER_CONFIG_DIR = (Resolve-Path ..\.jupyter\config)
    $env:JUPYTER_DATA_DIR = (Resolve-Path ..\.jupyter\data)
    $env:JUPYTER_RUNTIME_DIR = (Resolve-Path ..\.jupyter\runtime)
    $env:IPYTHONDIR = (Resolve-Path ..\.jupyter\ipython)
    $env:MPLCONFIGDIR = (Resolve-Path ..\.matplotlib)
    $env:KERAS_BACKEND = "torch"
    python -m jupyter notebook main.ipynb
    ```

4. Open the cells in [main.ipynb](./src/main.ipynb) and follow the markdown description. The training cells can take a long time to finish.

### Recommended training budget

The practical configuration used for the current training path is:

```text
population/actors: 10
generations: 10
MCST epochs: 2
MCST depth: 2
games per actor: 2, with both colors balanced
training horizon: 30 plies per game
```

The per-generation Stockfish benchmark is disabled during training because it does not affect selection and substantially increases runtime. Run the Stockfish evaluation cell after training instead.

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

The code expects the Stockfish executable at `Stockfish-master/src/stockfish.exe` on Windows, or `Stockfish-master/src/stockfish` on Linux/macOS.

For Windows, download the official x86-64 AVX2 build from the [Stockfish releases](https://github.com/official-stockfish/Stockfish/releases/latest), extract it, and place or rename the executable as:

```text
Stockfish-master/src/stockfish.exe
```

The Python code resolves this path from the repository location, so it no longer depends on the directory from which Jupyter was started.

For Linux/macOS, the source-build instructions are:

- download the repository: https://github.com/official-stockfish/Stockfish
- Execute the following commands in the terminal:
    ```bash 
        cd Stockfish-master/src
        make help
        make net
        make build ARCH=x86-64-modern
    ```
- Then move the `Stockfish-master` folder into the folder of this repository.

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
