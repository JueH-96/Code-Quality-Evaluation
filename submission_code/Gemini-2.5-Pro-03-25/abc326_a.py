# YOUR CODE HERE
import sys

# Read input from stdin
# Reads the two integers X and Y from a single line of input, separated by space.
x, y = map(int, sys.stdin.readline().split())

# Initialize a boolean variable to track if stairs are used.
# Default is False (assume elevator is used unless conditions for stairs are met).
use_stairs = False

# Check the conditions for using the stairs based on the problem description.

# Case 1: Moving up (destination floor Y is higher than starting floor X)
if y > x:
    # Calculate the number of floors moved up.
    floors_up = y - x
    # Check if the upward movement is 2 floors or less.
    if floors_up <= 2:
        use_stairs = True

# Case 2: Moving down (starting floor X is higher than destination floor Y)
# Since X != Y is guaranteed by the constraints, if not y > x, then x > y must hold.
elif x > y:
    # Calculate the number of floors moved down.
    floors_down = x - y
    # Check if the downward movement is 3 floors or less.
    if floors_down <= 3:
        use_stairs = True

# Print the final result based on whether the stairs are used.
if use_stairs:
    print("Yes")
else:
    print("No")