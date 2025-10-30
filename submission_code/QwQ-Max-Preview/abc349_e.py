import sys
from functools import lru_cache

# Read input
values = []
for _ in range(3):
    row = list(map(int, sys.stdin.readline().split()))
    values.extend(row)

@lru_cache(maxsize=None)
def is_terminal(state):
    # Check for three in a row, column, or diagonal
    lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for line in lines:
        a, b, c = line
        if state[a] == state[b] == state[c] and state[a] != 0:
            return 'Takahashi' if state[a] == 1 else 'Aoki'
    # Check if all cells are filled
    if 0 not in state:
        sum_red = sum(values[i] for i in range(9) if state[i] == 1)
        sum_blue = sum(values[i] for i in range(9) if state[i] == 2)
        return 'Takahashi' if sum_red > sum_blue else 'Aoki'
    return None

@lru_cache(maxsize=None)
def determine_winner(state):
    terminal_result = is_terminal(state)
    if terminal_result is not None:
        return terminal_result
    # Determine current player
    move_count = sum(1 for cell in state if cell != 0)
    current_player = 'Takahashi' if move_count % 2 == 0 else 'Aoki'
    available_moves = [i for i in range(9) if state[i] == 0]
    for move in available_moves:
        new_state = list(state)
        new_state[move] = 1 if current_player == 'Takahashi' else 2
        new_state = tuple(new_state)
        result = determine_winner(new_state)
        if result == current_player:
            return current_player
    # If no winning move, return the opponent
    return 'Aoki' if current_player == 'Takahashi' else 'Takahashi'

initial_state = tuple([0] * 9)
print(determine_winner(initial_state))