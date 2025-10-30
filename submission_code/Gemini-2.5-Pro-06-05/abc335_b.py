# YOUR CODE HERE
import sys

def solve():
    """
    Reads an integer N from stdin and prints all non-negative integer triples (x, y, z)
    such that x + y + z <= N, in lexicographical order.
    """
    try:
        # Read the integer N from a single line of standard input.
        N = int(sys.stdin.readline())
    except (IOError, ValueError):
        # In a contest setting, input is usually guaranteed to be valid.
        # This is here for robustness.
        return

    # To generate triples in lexicographical order, we use three nested loops.
    # The outer loop is for x, the middle for y, and the inner for z.

    # Iterate x from 0 to N.
    for x in range(N + 1):
        # For a given x, y can range from 0 up to N-x.
        for y in range(N + 1 - x):
            # For a given x and y, z can range from 0 up to N-x-y.
            for z in range(N + 1 - x - y):
                # The loops are constructed such that x + y + z <= N is always true.
                # Print the triple. The print() function separates arguments by spaces.
                print(x, y, z)

solve()