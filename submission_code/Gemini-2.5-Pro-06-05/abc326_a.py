# YOUR CODE HERE
import sys

def solve():
    """
    Reads two floor numbers and determines if stairs or an elevator is used.
    """
    try:
        # Read two integers X and Y from a single line of standard input
        X, Y = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # This handles cases where input is not as expected.
        return

    # Case 1: Moving up
    if Y > X:
        # He uses the stairs if moving up two floors or less.
        if Y - X <= 2:
            print("Yes")
        else:
            print("No")
    # Case 2: Moving down
    # This 'else' covers the Y < X case, since the problem states X != Y.
    else:
        # He uses the stairs if moving down three floors or less.
        if X - Y <= 3:
            print("Yes")
        else:
            print("No")

# It's a good practice to wrap the main logic in a function and call it.
solve()