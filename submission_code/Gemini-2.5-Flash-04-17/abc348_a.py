# YOUR CODE HERE
import sys

def solve():
    """
    Reads N, generates a string of length N where the i-th character
    is 'x' if i is a multiple of 3 and 'o' otherwise, and prints the string.
    The 1-based index i is used for the check.
    """
    # Read N from standard input
    N = int(sys.stdin.readline())

    # List to store the results
    results = []

    # Iterate from 1 to N (inclusive)
    for i in range(1, N + 1):
        # Check if the penalty kick number (i) is a multiple of 3
        if i % 3 == 0:
            # If it's a multiple of 3, it's a failure ('x')
            results.append('x')
        else:
            # Otherwise, it's a success ('o')
            results.append('o')

    # Join the list of characters into a single string and print it
    print("".join(results))

# Call the solve function to run the program
solve()