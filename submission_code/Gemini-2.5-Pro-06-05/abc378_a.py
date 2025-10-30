import sys
from collections import Counter

def solve():
    """
    Reads the ball colors, counts them, calculates the maximum number of pairs,
    and prints the result.
    """
    # Read the four integers representing ball colors from a single line of standard input.
    # The split() method separates the string by spaces, and map(int, ...) converts
    # each resulting string into an integer.
    try:
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle potential empty input or invalid format, though not expected by problem constraints.
        return

    # Use collections.Counter to efficiently count the frequency of each color.
    # For example, input '2 1 2 1' becomes a Counter object like {2: 2, 1: 2}.
    color_counts = Counter(A)

    # Initialize a variable to store the total number of operations (pairs).
    total_operations = 0

    # Iterate through the counts of each distinct color.
    # color_counts.values() gives us the counts (e.g., 2 and 2 for the example above).
    for count in color_counts.values():
        # For any given color, if we have 'count' balls, the number of pairs we can form
        # is 'count // 2' (integer division).
        # e.g., 2 balls -> 1 pair, 3 balls -> 1 pair, 4 balls -> 2 pairs.
        # We add the number of pairs for the current color to our total.
        total_operations += count // 2

    # Print the final result to standard output.
    print(total_operations)

# Execute the solution
solve()