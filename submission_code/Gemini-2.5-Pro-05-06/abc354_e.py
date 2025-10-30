import sys
import functools

# Global variables to store card data and N.
# This avoids passing them repeatedly in recursive calls, simplifying cache usage.
N_cards_global = 0
A_global = [] # Stores front values A_i
B_global = [] # Stores back values B_i

@functools.lru_cache(None) # Unbounded cache for memoization
def solve(mask):
    """
    Determines if the current player can win starting from the state 'mask'.
    A state is winning if there is at least one move to a losing state for the opponent.
    A state is losing if all moves lead to winning states for the opponent, or if no moves are possible.

    Args:
        mask (int): A bitmask representing the set of cards currently on the table.
                    If the i-th bit is set, card i is on the table.

    Returns:
        bool: True if the current player can win from this state, False otherwise.
    """

    # Iterate through all possible pairs of cards (idx1, idx2) on the table.
    # Card indices are 0 to N_cards_global - 1.
    for idx1 in range(N_cards_global):
        # Check if card idx1 is part of the current 'mask' (i.e., on the table)
        if not ((mask >> idx1) & 1):
            continue
        
        # Card idx1 is on the table. Now look for a second card, idx2.
        # To avoid duplicate pairs and self-pairing, idx2 starts from idx1 + 1.
        for idx2 in range(idx1 + 1, N_cards_global):
            # Check if card idx2 is part of the current 'mask'
            if not ((mask >> idx2) & 1):
                continue
            
            # Both cards idx1 and idx2 are on the table.
            # Check if they form a removable pair.
            # A pair is removable if their front values (A) match or their back values (B) match.
            can_remove_this_pair = False
            if A_global[idx1] == A_global[idx2] or B_global[idx1] == B_global[idx2]:
                can_remove_this_pair = True
            
            if can_remove_this_pair:
                # This pair can be removed. This constitutes a valid move.
                # Calculate the mask for the state after removing this pair.
                # Removing cards idx1 and idx2 means flipping their corresponding bits in the mask.
                mask_after_move = mask ^ (1 << idx1) ^ (1 << idx2)
                
                # The current player makes this move. The turn passes to the opponent,
                # who will play from 'mask_after_move'.
                # If the opponent *loses* from 'mask_after_move' (i.e., solve(mask_after_move) is False),
                # then the current move is a winning move for the current player.
                if not solve(mask_after_move):
                    # Found a winning strategy for the current player.
                    return True 
    
    # If the function reaches this point, it means:
    # 1. No moves were possible (e.g., mask is 0, or cards remaining don't form any valid pairs).
    # 2. All possible moves lead to states from which the opponent wins.
    # In all these scenarios, the current player loses from state 'mask'.
    return False

def main():
    # Make N_cards_global, A_global, B_global accessible for modification
    global N_cards_global, A_global, B_global
    
    N_cards_global = int(sys.stdin.readline())
    
    A_global = [0] * N_cards_global
    B_global = [0] * N_cards_global
    
    for i in range(N_cards_global):
        ai, bi = map(int, sys.stdin.readline().split())
        A_global[i] = ai
        B_global[i] = bi
        
    # The game starts with all N cards on the table.
    # The initial mask has the N least significant bits set.
    initial_mask = (1 << N_cards_global) - 1
    
    # Takahashi is the first player. Determine if Takahashi wins.
    if solve(initial_mask):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == '__main__':
    main()