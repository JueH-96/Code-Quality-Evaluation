import sys

def solve():
    """
    This function contains the main logic to solve the problem for multiple test cases.
    It reads from standard input and writes to standard output.
    """
    # An alias for sys.stdin.readline for faster I/O and cleaner code.
    input = sys.stdin.readline

    # Read the number of test cases.
    try:
        t = int(input())
    except (ValueError, IndexError):
        # Handle cases with no input, e.g., during local testing.
        t = 0

    # Process each test case.
    for _ in range(t):
        # Read the length of the strip (n) and the operation length (k).
        try:
            n, k = map(int, input().split())
            # Read the strip configuration as a string.
            s = input().strip()
        except (ValueError, IndexError):
            # Skip if a test case input is malformed.
            continue
        
        operations = 0
        i = 0  # Pointer to scan the strip.
        
        while i < n:
            # If the current cell is black, an operation is required.
            if s[i] == 'B':
                operations += 1
                # The greedy operation covers the block of k cells starting at i.
                # After this, we can jump our pointer k steps forward.
                i += k
            else:
                # If the cell is white, just move to the next cell.
                i += 1
        
        # Print the minimum number of operations for the current test case.
        print(operations)

# Run the solution.
solve()