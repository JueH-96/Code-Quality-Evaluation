# YOUR CODE HERE
import sys

# Function to solve a single test case
def solve():
    # Read the input values for n (length of the paper) and k (operation length)
    n, k = map(int, sys.stdin.readline().split())
    # Read the strip of paper as a string consisting of 'B' (black) or 'W' (white)
    s = sys.stdin.readline().strip()

    # Initialize the counter for the minimum number of operations
    operations = 0
    # Initialize the index to traverse the paper from left to right
    i = 0

    # Iterate through the strip. The loop continues as long as the current index 'i' is within the strip bounds [0, n-1].
    # We only need to process up to index n-1.
    while i < n:
        # Check if the cell at the current index 'i' is black.
        if s[i] == 'B':
            # If the cell is black, it means we have found the leftmost black cell that is not yet covered by a previous operation.
            # We MUST perform an operation to cover this black cell.
            # We increment the count of operations needed.
            operations += 1
            # According to the greedy strategy, we apply an operation starting at index 'i'.
            # This operation covers a segment of 'k' consecutive cells, specifically from index 'i' to index 'i + k - 1'.
            # All cells within this covered segment [i, i+k-1] become white.
            # Since these 'k' cells are now white, we do not need to check them individually in the next iterations.
            # The next cell we need to examine is the one immediately following this covered segment, which is at index 'i + k'.
            # So, we jump the index 'i' forward by 'k' steps.
            i += k
        else:
            # If the cell at the current index s[i] is 'W' (white), it is already in the desired state.
            # It does not require any operation to be made white.
            # We simply move to the next cell to continue scanning the strip for black cells.
            # We increment the index 'i' by 1.
            i += 1

    # After the 'while' loop finishes, the index 'i' is greater than or equal to 'n'.
    # This indicates that we have traversed the entire strip from left to right.
    # Every black cell encountered was addressed by performing an operation covering it and the subsequent k-1 cells.
    # Therefore, all black cells in the original strip are now conceptually covered and turned white.
    # The total number of operations performed is stored in the 'operations' variable.
    # Print the result for the current test case to standard output.
    print(operations)

# Read the total number of test cases from the first line of the input.
t = int(sys.stdin.readline())

# Process each test case one by one.
for _ in range(t):
    # Call the solve function to handle the current test case.
    solve()