import sys

# It's good practice in competitive programming to use fast I/O.
# We will use sys.stdin.readline for reading input efficiently.
# For this problem's constraints, standard input() would also be acceptable.
input = sys.stdin.readline

def solve():
    """
    Reads the grid dimensions and content, then calculates the minimum operations.
    """
    try:
        # Read grid dimensions H, W and sequence length K
        H, W, K = map(int, input().split())
        # Read the grid as a list of strings
        S = [input().strip() for _ in range(H)]
    except (ValueError, IndexError):
        # This handles cases where input might be empty, common in some online judges.
        return

    # Initialize the minimum operations to a value larger than any possible result.
    min_ops = float('inf')

    # --- Horizontal Check ---
    # A horizontal sequence of K 'o's is possible only if K is not larger than the grid width.
    if K <= W:
        # Iterate through each row of the grid
        for i in range(H):
            # Use a sliding window of size K to check all horizontal subsequences.
            # Initialize counts for '.' and 'x' characters.
            dots_count = 0
            crosses_count = 0

            # Set up the counts for the initial window (from column 0 to K-1)
            for j in range(K):
                if S[i][j] == '.':
                    dots_count += 1
                elif S[i][j] == 'x':
                    crosses_count += 1
            
            # If this first window contains no 'x's, it's a candidate.
            # The number of operations is the number of '.'s to change.
            if crosses_count == 0:
                min_ops = min(min_ops, dots_count)

            # Slide the window one position to the right across the rest of the row
            for j in range(K, W):
                # Add the character entering the window from the right
                if S[i][j] == '.':
                    dots_count += 1
                elif S[i][j] == 'x':
                    crosses_count += 1
                
                # Remove the character leaving the window from the left
                left_char = S[i][j - K]
                if left_char == '.':
                    dots_count -= 1
                elif left_char == 'x':
                    crosses_count -= 1
                
                # If the new window is a valid candidate, update the minimum operations
                if crosses_count == 0:
                    min_ops = min(min_ops, dots_count)

    # --- Vertical Check ---
    # A vertical sequence of K 'o's is possible only if K is not larger than the grid height.
    if K <= H:
        # Iterate through each column of the grid
        for j in range(W):
            # Use a sliding window, similar to the horizontal check
            dots_count = 0
            crosses_count = 0

            # Initialize the counts for the first vertical window
            for i in range(K):
                if S[i][j] == '.':
                    dots_count += 1
                elif S[i][j] == 'x':
                    crosses_count += 1
            
            if crosses_count == 0:
                min_ops = min(min_ops, dots_count)

            # Slide the window one position down the column
            for i in range(K, H):
                # Add the character entering from the bottom
                if S[i][j] == '.':
                    dots_count += 1
                elif S[i][j] == 'x':
                    crosses_count += 1
                
                # Remove the character leaving from the top
                top_char = S[i - K][j]
                if top_char == '.':
                    dots_count -= 1
                elif top_char == 'x':
                    crosses_count -= 1
                
                if crosses_count == 0:
                    min_ops = min(min_ops, dots_count)

    # --- Final Output ---
    # If min_ops was never updated, no valid sequence was found.
    if min_ops == float('inf'):
        print(-1)
    else:
        print(min_ops)

solve()