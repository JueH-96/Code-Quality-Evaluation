# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    """
    Reads two integers A and B from standard input,
    determines if the squares A and B are horizontally adjacent on a 3x3 board,
    and prints "Yes" or "No" to standard output.
    The board layout is assumed to be:
    1 2 3
    4 5 6
    7 8 9
    """
    # Read the two integers A and B from stdin, separated by space
    line = sys.stdin.readline().split()
    a = int(line[0])
    b = int(line[1])

    # Check the conditions for horizontal adjacency:
    # Condition 1: B must be exactly one greater than A (B = A + 1).
    #              This ensures they are consecutive numbers. If B is not A+1,
    #              they cannot be adjacent in any standard grid sense (horizontally or vertically).
    # Condition 2: A must not be in the rightmost column of the 3x3 grid.
    #              The numbers in the rightmost column are 3 and 6.
    #              (Number 9 is also in the rightmost column, but since A < B <= 9, A cannot be 9).
    #              If A is 3 or 6, then A+1 (which would be B if condition 1 holds)
    #              is on the start of the next row, not horizontally adjacent to A.
    if b == a + 1 and a != 3 and a != 6:
        # If both conditions are true, the squares A and B are horizontally adjacent.
        print("Yes")
    else:
        # If either condition is false, they are not horizontally adjacent.
        print("No")

# Call the solve function to execute the logic
solve()

# YOUR CODE HERE