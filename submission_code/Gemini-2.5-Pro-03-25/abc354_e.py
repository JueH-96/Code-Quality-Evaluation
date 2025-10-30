# YOUR CODE HERE
import sys

# Set higher recursion depth limit if necessary.
# The maximum recursion depth for N=18 is N/2 = 9, which is typically well within
# Python's default limit (often 1000). However, setting it explicitly can prevent
# issues in environments with lower defaults or for larger N if constraints change.
try:
    # Set a large recursion depth limit, e.g., 200000
    # This accommodates potentially deep recursion paths, although unlikely for N=18.
    sys.setrecursionlimit(200000) 
except OverflowError:
    # Handle cases where setting recursion limit might fail (e.g. restricted environments)
    # or is excessively large. The program might still work if default limit is sufficient.
    pass


def solve():
    """
    Solves the card game problem using recursion with memoization (dynamic programming on subsets).
    Determines the winner assuming optimal play from both players.
    """
    N = int(sys.stdin.readline())
    cards = []
    for i in range(N):
        # Read card values A_i, B_i
        cards.append(list(map(int, sys.stdin.readline().split()))) 

    # Precompute all pairs of card indices (i, j) with i < j such that 
    # card i and card j can be removed together according to the game rules.
    # A pair (i, j) can be removed if A_i == A_j or B_i == B_j.
    possible_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            A_i, B_i = cards[i]
            A_j, B_j = cards[j]
            if A_i == A_j or B_i == B_j:
                possible_pairs.append((i, j))

    # Memoization dictionary to store results for game states (represented by masks).
    # Key: integer mask representing the set of cards remaining.
    # Value: Boolean, True if the current player can win from this state, False otherwise.
    memo = {} 

    # Recursive function to determine if the current player can win from state 'mask'.
    def can_win(mask):
        """
        Determines if the current player can force a win starting from the state represented by 'mask'.
        Uses memoization to avoid redundant computations.
        Returns True if the current player can win, False otherwise.
        """
        # Check if the result for this state (mask) is already computed and stored in memo.
        if mask in memo:
            return memo[mask]
        
        # Base case: If mask is 0, it means no cards are left. 
        # The current player cannot make a move and thus loses.
        # This check is implicitly handled by the loop logic, but can be explicit for clarity:
        # if mask == 0:
        #     memo[mask] = False
        #     return False

        # Iterate through all precomputed pairs (i, j) that could potentially be removed.
        for i, j in possible_pairs:
            # Check if both cards i and j are present in the current state 'mask'.
            # This is done using bitwise AND operations.
            # (mask & (1 << i)) is non-zero if the i-th bit of mask is 1.
            # (mask & (1 << j)) is non-zero if the j-th bit of mask is 1.
            if (mask & (1 << i)) and (mask & (1 << j)):
                # If both cards i and j are present, this pair represents a valid move.
                # Calculate the next state's mask by removing cards i and j.
                # This is done using bitwise XOR: flips the bits at positions i and j from 1 to 0.
                new_mask = mask ^ (1 << i) ^ (1 << j) 
                
                # Recursively call `can_win` for the `new_mask`. This determines if the *opponent*
                # can win from the resulting state.
                # If `can_win(new_mask)` returns False, it means the opponent *cannot* win from `new_mask`.
                # This implies that `new_mask` is a losing state for the player whose turn it is.
                # Therefore, the current player *can* win by making this move (i, j).
                if not can_win(new_mask):
                    # Found a winning move. Store True in memo for the current state 'mask'.
                    memo[mask] = True
                    # Return True immediately as we only need one winning move.
                    return True
        
        # If the loop completes without finding any winning move:
        # This means either there were no valid moves possible from 'mask' (either mask was 0 initially,
        # or no pair (i,j) among the present cards satisfies the removal condition),
        # OR all possible moves lead to states from which the opponent can win.
        # In either scenario, the current state 'mask' is a losing state for the current player.
        # Store False in memo for the current state 'mask'.
        memo[mask] = False
        return False

    # The game starts with all N cards present. The initial state is represented by a mask
    # where all N bits (from 0 to N-1) are set to 1.
    initial_mask = (1 << N) - 1
    
    # Call `can_win` for the initial state to determine if the first player (Takahashi) can win.
    if can_win(initial_mask):
        # If can_win returns True, Takahashi wins.
        print("Takahashi")
    else:
        # If can_win returns False, Aoki (the second player) wins.
        print("Aoki")

# Call the solve function to execute the main logic of the program.
solve()