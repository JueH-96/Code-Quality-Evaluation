import sys

# Global variables to be accessed by the recursive function.
# Using global variables for N, cards, and memo can sometimes be slightly faster
# than passing them as arguments in recursive calls, especially in competitive programming.
N = 0
cards = []
memo = {} # Dictionary for memoization: mask -> True (win) or False (lose)

def solve(mask):
    """
    Determines if the current player can win from the given state (mask).
    A state is represented by a bitmask where the i-th bit is 1 if card i is present.
    """
    # If the result for this mask is already computed, return it
    if mask in memo:
        return memo[mask]

    # Assume the current player loses unless a winning move is found
    # Iterate over all possible pairs of cards (i, j) in the current mask
    # We use nested loops to select two distinct cards i and j.
    for i in range(N):
        # Check if card i is present in the current mask
        # (mask >> i) & 1 checks if the i-th bit is set
        if not ((mask >> i) & 1):
            continue # Card i is not on the table, skip

        # Start j from i + 1 to avoid duplicate pairs (i,j) and (j,i) and self-pairs (i,i)
        for j in range(i + 1, N):
            # Check if card j is present in the current mask
            if not ((mask >> j) & 1):
                continue # Card j is not on the table, skip

            # Check if cards i and j can be removed according to the rules:
            # either their front numbers (A) are the same or their back numbers (B) are the same.
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                # If they can be removed, form the next mask by removing cards i and j.
                # (1 << i) creates a mask with only the i-th bit set.
                # XORing with these masks flips the bits for i and j, effectively removing them.
                next_mask = mask ^ (1 << i) ^ (1 << j)

                # Recursively call solve for the next state.
                # If the *next* player (opponent) loses from next_mask,
                # then the current player wins by making this move.
                if not solve(next_mask):
                    memo[mask] = True # Current player found a winning move
                    return True       # Found a winning move, so current player wins

    # If no winning move was found after checking all possibilities,
    # then the current player cannot win from this state.
    memo[mask] = False
    return False

def main():
    global N, cards # Declare global to modify these variables within main()

    # Read N from standard input
    N = int(sys.stdin.readline())

    # Read card data (A_i, B_i) for each of the N cards
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        cards.append((A, B))

    # The initial state is when all N cards are on the table.
    # This mask has all N bits set to 1. For N=5, it's (1 << 5) - 1 = 31 (binary 11111).
    initial_mask = (1 << N) - 1

    # Determine if Takahashi (the first player) wins from the initial state
    if solve(initial_mask):
        sys.stdout.write("Takahashi
")
    else:
        sys.stdout.write("Aoki
")

if __name__ == "__main__":
    main()