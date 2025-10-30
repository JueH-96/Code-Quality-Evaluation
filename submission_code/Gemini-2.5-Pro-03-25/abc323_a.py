# YOUR CODE HERE
import sys

def solve():
    """
    Reads a 16-character binary string S and checks if characters
    at even positions (2, 4, ..., 16) are all '0'.
    Prints "Yes" if true, "No" otherwise.
    """
    # Read the input string from standard input
    S = sys.stdin.readline().strip()

    # Check if the length constraint is met (although guaranteed by problem statement)
    if len(S) != 16:
        # Handle error or assume valid input based on problem constraints
        # For this problem, we assume input is always valid.
        pass

    # We need to check positions 2, 4, 6, 8, 10, 12, 14, 16.
    # In 0-based indexing, these correspond to indices 1, 3, 5, 7, 9, 11, 13, 15.

    # We can iterate through these specific indices and check the character.
    # Assume the condition is true initially.
    all_required_zeros = True

    # Loop through the relevant indices: 1, 3, 5, ..., 15
    # range(start, stop, step) generates numbers from start up to (but not including) stop.
    for i in range(1, 16, 2):
        # Access the character at the current index i
        if S[i] == '1':
            # If we find a '1' at any of these positions, the condition is false.
            all_required_zeros = False
            # No need to check further, we can break the loop.
            break

    # Print the final result based on the flag.
    if all_required_zeros:
        print("Yes")
    else:
        print("No")

# Call the solve function to execute the logic
solve()