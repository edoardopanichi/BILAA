## Sources
- https://towardsdatascience.com/building-a-chess-ai-that-learns-from-experience-5cff953b6784
- https://arxiv.org/pdf/1711.08337.pdf
- https://medium.com/@ishaan.gupta0401/monte-carlo-tree-search-application-on-chess-5573fc0efb75 
- Random ELO: https://chess.stackexchange.com/questions/6508/what-would-be-the-elo-of-a-computer-program-that-plays-at-random

## Possible modification needed:
- self.v in MCST might be calculated as average of value of the state Si. Check notes on iPad
- ROLLOUT procedure might be substituted with a NN whose weights are picked with the EA 

## Documentation: 
- Stockfish library: https://pypi.org/project/stockfish/

## Install stockfish:
- download the repository: https://github.com/official-stockfish/Stockfish
- ```bash 
    cd src
    make help
    make net
    make build ARCH=x86-64-modern
    ```
- then if you move the folder Stockfish-master into the folder of this repository, the code will work correctly. Otherwise you need to modify the absolute path in where the stockfish engine is initialized. For example: Stockfish(path="/Users/edoardo/Downloads/Stockfish-master/src/stockfish")

## To Do:
- check the EA functions for possible bugs
- correct the chess_function set_ELO_and_skills to just set_skills
- create a new way to evaluate an agent based on stockfish evaluation (percentage of guesses that belong to the top 3 moves of stockfish)