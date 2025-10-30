# YOUR CODE HERE
import sys

def solve():
    """
    Reads an integer X from standard input, finds the positive integer N
    such that N! = X, and prints N to standard output.
    It is guaranteed that such an N exists and is unique, and X >= 2.
    """
    # Read the input integer X from standard input
    x = int(sys.stdin.readline())

    # Initialize n and factorial starting from 1! = 1
    n = 1
    current_factorial = 1

    # Since X >= 2, we know that N must be at least 2.
    # We will iteratively compute factorials 2!, 3!, ...
    # until the computed factorial equals X.

    # We can use a loop that continues as long as the current factorial is less than X.
    # Python's integers handle arbitrary precision, so overflow is not an issue
    # for the given constraints (X <= 3 * 10^18, which means N <= 20).
    while current_factorial < x:
        # Increment n to consider the next integer for the factorial calculation
        n += 1
        # Update the factorial: current_factorial becomes n!
        # (calculated as (n-1)! * n)
        current_factorial *= n

    # After the loop terminates, current_factorial will be >= X.
    # The problem statement guarantees that there is exactly one positive integer N
    # such that N! = X. This implies that the loop must stop exactly when
    # current_factorial becomes equal to X.
    # Therefore, the value of 'n' at this point is the answer we are looking for.
    
    # Print the resulting value of n to standard output
    print(n)

# Execute the solve function to run the logic
solve()