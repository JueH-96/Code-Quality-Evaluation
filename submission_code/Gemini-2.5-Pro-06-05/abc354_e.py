import sys

def solve():
    """
    Main function to solve the problem.
    Reads input, determines the winner of the card game using recursion with
    memoization, and prints the result.
    """
    # The default recursion limit is sufficient as the max depth is N/2 = 9.

    # Read problem size and card data from standard input
    try:
        N = int(sys.stdin.readline())
        cards = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    except (IOError, ValueError):
        # Handles empty input on some platforms
        return

    # Unpack card data into separate lists for front (A) and back (B) numbers
    A = [card[0] for card in cards]
    B = [card[1] for card in cards]

    # Memoization table to cache results of subproblems.
    memo = {}

    def can_win(mask):
        """
        Determines if the current player can force a win from the game state
        represented by the bitmask 'mask'.
        
        Args:
            mask (int): A bitmask where the i-th bit is 1 if the i-th card is on the table.

        Returns:
            bool: True if the current state is a winning position, False otherwise.
        """
        # If this state has been evaluated before, return the cached result.
        if mask in memo:
            return memo[mask]

        # Iterate through all distinct pairs of cards (i, j) on the table.
        for i in range(N):
            # If card i is not on the table, continue.
            if not ((mask >> i) & 1):
                continue
            
            for j in range(i + 1, N):
                # If card j is not on the table, continue.
                if not ((mask >> j) & 1):
                    continue
                
                # Check if removing cards (i, j) is a valid move.
                if A[i] == A[j] or B[i] == B[j]:
                    # The move is valid. Calculate the resulting state.
                    next_mask = mask ^ (1 << i) ^ (1 << j)
                    
                    # If the opponent cannot win from the next state, it means we
                    # have found a winning move.
                    if not can_win(next_mask):
                        # Cache and return True, as this is a winning state.
                        memo[mask] = True
                        return True

        # If the loop completes, no winning move was found.
        # This implies all moves lead to states where the opponent can win.
        # Therefore, this is a losing state.
        memo[mask] = False
        return False

    # The game starts with all N cards on the table.
    # The initial mask is (1<<N) - 1, which has N bits set to 1.
    initial_mask = (1 << N) - 1

    # Takahashi plays first. The winner is determined by whether the
    # initial state is a winning position for the first player.
    if can_win(initial_mask):
        print("Takahashi")
    else:
        print("Aoki")

# Execute the main logic.
solve()