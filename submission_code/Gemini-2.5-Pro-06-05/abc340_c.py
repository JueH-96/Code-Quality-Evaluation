# YOUR CODE HERE
import sys

# It's a good practice for recursive solutions on some platforms to increase
# the recursion limit, although for this problem's constraints (N <= 10^17),
# the recursion depth is small (~log2(N)), so the default limit is sufficient.
# sys.setrecursionlimit(2000)

# A dictionary to store the results of subproblems (memoization)
memo = {}

def calculate_total_cost(n):
    """
    Recursively calculates the total cost to break down an integer n into
    numbers less than 2, using memoization to store and retrieve results
    of subproblems.
    """
    # Base case: If n is less than 2, no more operations are possible, so the cost is 0.
    if n < 2:
        return 0

    # If the cost for n has already been computed, return the stored value.
    if n in memo:
        return memo[n]

    # The problem's total cost can be found using the recurrence relation:
    # Cost(n) = n + Cost(floor(n/2)) + Cost(ceil(n/2))
    
    # Calculate floor(n/2) using integer division.
    floor_val = n // 2
    
    # Calculate ceil(n/2). For a non-negative integer n,
    # this is equivalent to n - floor(n/2).
    ceil_val = n - floor_val

    # The total cost is the cost of the current operation (n) plus the
    # costs for the two resulting numbers.
    total_cost = n + calculate_total_cost(floor_val) + calculate_total_cost(ceil_val)

    # Store the computed result in the memoization dictionary before returning.
    memo[n] = total_cost
    
    return total_cost

def main():
    """
    Reads the input integer N, solves the problem, and prints the result.
    """
    # Read the single integer N from standard input.
    try:
        N = int(sys.stdin.readline())
    except (IOError, ValueError):
        # In case of malformed input, exit gracefully.
        return

    # Calculate the total cost for N using the recursive function.
    result = calculate_total_cost(N)
    
    # Print the final result to standard output.
    print(result)

if __name__ == "__main__":
    main()