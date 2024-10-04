# Tic Tac Toe with Adversarial Search AI 

A Tic Tac Toe game implemented in Python using Tkinter for the graphical user interface. The game allows two players to compete against each other or against an AI opponent, with multiple AI algorithms to choose from.

## Features

- **Interactive GUI**: Built with Tkinter for a user-friendly interface.
- **AI Opponent**: Choose between different AI algorithms for a challenging gameplay experience.
- **Customizable Board Size**: Supports different dimensions and winning conditions.
- **Real-time Updates**: The game updates dynamically as players take turns.

## Technologies

- **Python 3.x**
- **Tkinter**: For GUI interface development.
- **Custom AI Algorithms**: MinMax, AlphaBeta, Expectimax, and Random player.

## Installation

To run the project, make sure you have Python 3.x installed on your machine or in your vscode environment.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tic-tac-toe.git
   cd tic-tac-toe


## Usage

To start the game, run the main script:

   ```bash
   python tic-tac-toe.py [height] [width] [k-match]


   height: The number of rows in the Tic Tac Toe grid (default is 3).
   width: The number of columns in the Tic Tac Toe grid (default is 3).
   k-match: The number of symbols in a row needed to win (default is 3).

   Example:

   bash
   Copy code
   python main.py 4 4 4
   Gameplay
   Players take turns clicking on empty buttons to place their symbol (X or O).
   The game automatically checks for victory conditions after each move.
   The AI makes its move after the player’s turn, based on the selected algorithm.
AI Algorithms
The game includes the following AI algorithms:

Random: The AI chooses a random empty position.
MinMax: A simple minimax algorithm that aims to minimize the player's chances of winning.
AlphaBeta: An optimized version of the MinMax algorithm that reduces the number of nodes evaluated.
Expectimax: Considers the expected utility of moves, making it a more sophisticated choice for AI opponents.
You can select the desired algorithm from the dropdown menu in the UI.

Game Reset and Exit
Reset: Click the "Reset" button to start a new game without closing the application.
Exit: Click the "Exit" button to close the game window.
