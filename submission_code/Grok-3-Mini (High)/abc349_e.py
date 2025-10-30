import sys
from typing import Tuple, Dict

# Read input and store A as a flat tuple
data = list(map(int, sys.stdin.read().split()))
A_flat = tuple(data)

# Define has_three_in_row function
def has_three_in_row(occ: Tuple[int, ...], color: int) -> bool:
    # Check rows
    for i in range(3):
        if all(occ[i * 3 + j] == color for j in range(3)):
            return True
    # Check columns
    for j in range(3):
        if all(occ[i * 3 + j] == color for i in range(3)):
            return True
    # Check diagonals
    if all(occ[i * 3 + i] == color for i in range(3)):
        return True
    if all(occ[i * 3 + (2 - i)] == color for i in range(3)):
        return True
    return False

# Memoization dictionary
memo: Dict[Tuple[Tuple[int, ...], int], int] = {}

# Evaluate state function with memoization
def eval_state(occ: Tuple[int, ...], turn: int) -> int:
    key = (occ, turn)
    if key in memo:
        return memo[key]
    
    empty_indices = [idx for idx in range(9) if occ[idx] == 0]
    
    if turn == 1:  # Takahashi's turn
        max_outcome = -1
        for idx in empty_indices:
            new_occ_list = list(occ)
            new_occ_list[idx] = 1  # Takahashi's color
            new_occ_tuple = tuple(new_occ_list)
            
            if has_three_in_row(new_occ_tuple, 1):  # Takahashi wins immediately
                outcome = 1
            elif all(cell != 0 for cell in new_occ_tuple):  # All cells taken
                score_T = sum(A_flat[i] for i in range(9) if new_occ_tuple[i] == 1)
                score_A = sum(A_flat[i] for i in range(9) if new_occ_tuple[i] == 2)
                outcome = 1 if score_T > score_A else -1
            else:  # Game continues with Aoki's turn
                outcome = eval_state(new_occ_tuple, 2)
            
            if outcome > max_outcome:
                max_outcome = outcome
        
        memo[key] = max_outcome
        return max_outcome
    
    else:  # Aoki's turn
        min_outcome = 1
        for idx in empty_indices:
            new_occ_list = list(occ)
            new_occ_list[idx] = 2  # Aoki's color
            new_occ_tuple = tuple(new_occ_list)
            
            if has_three_in_row(new_occ_tuple, 2):  # Aoki wins immediately
                outcome = -1
            elif all(cell != 0 for cell in new_occ_tuple):  # All cells taken
                score_T = sum(A_flat[i] for i in range(9) if new_occ_tuple[i] == 1)
                score_A = sum(A_flat[i] for i in range(9) if new_occ_tuple[i] == 2)
                outcome = 1 if score_T > score_A else -1
            else:  # Game continues with Takahashi's turn
                outcome = eval_state(new_occ_tuple, 1)
            
            if outcome < min_outcome:
                min_outcome = outcome
        
        memo[key] = min_outcome
        return min_outcome

# Initial state
initial_occ = (0,) * 9  # All cells unoccupied
result = eval_state(initial_occ, 1)  # Takahashi starts

# Output the winner
if result == 1:
    print("Takahashi")
else:
    print("Aoki")