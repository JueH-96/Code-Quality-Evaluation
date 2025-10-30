# YOUR CODE HERE
import sys

# Function to calculate N_even(L, R): count of even integers in [L, R]
def count_even(L, R):
    """Counts the number of even integers in the inclusive range [L, R].
    
    Args:
        L: The lower bound of the range.
        R: The upper bound of the range.

    Returns:
        The count of even integers in [L, R]. Returns 0 if L > R.
    """
    if L > R:
        return 0
    # The number of even integers less than or equal to X is floor(X/2).
    # The number of even integers less than L is the number of even integers less than or equal to L-1, which is floor((L-1)/2).
    # The number of even integers in [L, R] is (count <= R) - (count < L).
    return R // 2 - (L - 1) // 2

# Function to calculate N_odd(L, R): count of odd integers in [L, R]
def count_odd(L, R):
    """Counts the number of odd integers in the inclusive range [L, R].

    Args:
        L: The lower bound of the range.
        R: The upper bound of the range.

    Returns:
        The count of odd integers in [L, R]. Returns 0 if L > R.
    """
    if L > R:
        return 0
    # The number of odd integers less than or equal to X is floor((X+1)/2).
    # The number of odd integers less than L is the number of odd integers less than or equal to L-1, which is floor(((L-1)+1)/2) = floor(L/2).
    # The number of odd integers in [L, R] is (count <= R) - (count < L).
    return (R + 1) // 2 - L // 2

def solve():
    """Reads input coordinates, calculates minimum toll based on pathfinding logic, and prints the result."""
    # Read start coordinates (Sx, Sy) and target coordinates (Tx, Ty)
    sx, sy = map(int, sys.stdin.readline().split())
    tx, ty = map(int, sys.stdin.readline().split())

    # Case 1: Start and target y-coordinates are the same (sy == ty).
    # The path involves only horizontal movement along the row y = sy.
    # The cost depends on the parity of squares crossed and the parity of y.
    if sy == ty:
        y = sy
        # If x-coordinates are also the same, no movement needed, cost is 0.
        if sx == tx:
            print(0)
        # If start x < target x, movement is Right.
        elif sx < tx:
            # Path involves moving Right. A step Right from (i, y) to (i+1, y) occurs for i from sx to tx-1.
            # The cost of moving Right from (i, y) is 1 if i+y is odd.
            # We need to sum this cost over the relevant range of i.
            L = sx
            R = tx - 1 # The range of starting i-coordinates for Right moves.
            
            if y % 2 == 0: # If y is even, cost is 1 if i is odd. Total cost H is the count of odd i's in [L, R].
                H = count_odd(L, R)
            else: # If y is odd, cost is 1 if i is even. Total cost H is the count of even i's in [L, R].
                H = count_even(L, R)
            print(H)

        # If start x > target x, movement is Left.
        else: # sx > tx
            # Path involves moving Left. A step Left from (i, y) to (i-1, y) occurs for i from sx down to tx+1.
            # The cost of moving Left from (i, y) is 1 if i+y is even.
            # The range of i values (from which a Left move starts) is [tx+1, sx].
            L = tx + 1
            R = sx
            
            if y % 2 == 0: # If y is even, cost is 1 if i is even. Total cost H is the count of even i's in [L, R].
                 H = count_even(L, R)
            else: # If y is odd, cost is 1 if i is odd. Total cost H is the count of odd i's in [L, R].
                 H = count_odd(L, R)
            print(H)

    # Case 2: Start and target y-coordinates are different (sy != ty).
    # Vertical movement is necessary. The path must cross abs(ty - sy) horizontal tile boundaries (lines y=k for integers k).
    # Each crossing of a horizontal boundary y=k costs 1 toll, regardless of the x-coordinate.
    # The total minimum cost due to vertical movement is abs(ty - sy).
    # Crucially, it's possible to construct a path such that all horizontal movements incur 0 toll.
    # This can be achieved by potentially making a vertical move (Up or Down) before a horizontal move 
    # to ensure that the horizontal move starts from a square (i, j) with the required parity (i+j odd for Right, i+j even for Left) to achieve zero cost.
    # Since vertical moves are necessary anyway (costing abs(ty - sy) in total), and these moves allow optimizing horizontal steps to cost 0,
    # the minimum total cost is determined solely by the vertical displacement.
    else: 
        print(abs(ty - sy))

# Execute the solve function to run the program logic.
solve()