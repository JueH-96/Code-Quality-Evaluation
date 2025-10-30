# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input sequence, computes the sequence B, and prints it.
    """
    # Read N, the number of elements in the sequence.
    try:
        n_str = sys.stdin.readline()
        if not n_str:
            return  # Handle empty input at EOF
        n = int(n_str)
        
        # Read the sequence A of N space-separated integers.
        a = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # Exit gracefully if input is malformed.
        return

    # A dictionary to store the most recent 1-based index for each number.
    # Key: A number from the sequence `a`.
    # Value: The most recent 1-based position where this number was found.
    last_seen_pos = {}

    # The list to hold the elements of the resulting sequence B.
    b = []

    # Iterate through the input sequence `a`. `i` is the 0-based index.
    for i, val in enumerate(a):
        # Look up the last seen position of the current value `val`.
        # The .get() method is perfect here: it returns the stored position if `val`
        # is in the dictionary, otherwise it returns the default value, -1.
        prev_pos = last_seen_pos.get(val, -1)
        
        # Append the found position to our result list.
        b.append(prev_pos)

        # Update the dictionary with the current position of `val`.
        # The problem requires 1-based indexing for positions, so we use `i + 1`.
        last_seen_pos[val] = i + 1

    # Print the final sequence B. The `*` operator unpacks the list `b`,
    # and `print` separates the items with spaces by default.
    print(*b)

# Execute the main logic.
solve()