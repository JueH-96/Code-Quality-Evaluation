import sys

# Memoization dictionary to store calculated costs
# Base case: cost to break down 1 is 0, as it's already a leaf and no operation is performed on it.
memo = {1: 0}

def calculate_total_cost(n):
    """
    Calculates the total amount of money Takahashi will have paid
    to break down the integer n into ones.
    Uses memoization to avoid redundant computations.

    Args:
        n (int): The current integer on the blackboard to process.

    Returns:
        int: The total cost incurred for processing n and its descendants.
    """
    # If the cost for 'n' has already been computed, return it from memoization.
    if n in memo:
        return memo[n]

    # The current operation on 'n' costs 'n' yen.
    current_cost = n

    # Determine how 'n' splits and recursively calculate costs for its children.
    if n % 2 == 0:
        # If n is even, it splits into n/2 and n/2.
        # Both children are the same, so we recursively sum their costs twice.
        current_cost += 2 * calculate_total_cost(n // 2)
    else:
        # If n is odd, it splits into floor(n/2) and ceil(n/2).
        # floor(n/2) is n // 2 in integer division.
        # ceil(n/2) is n // 2 + 1 for positive odd n.
        current_cost += calculate_total_cost(n // 2) + calculate_total_cost(n // 2 + 1)
    
    # Store the computed cost in the memoization dictionary before returning.
    memo[n] = current_cost
    return current_cost

# Main execution block
if __name__ == '__main__':
    # Read the initial integer N from standard input.
    N = int(sys.stdin.readline())
    
    # Calculate the total cost starting from N.
    total_paid = calculate_total_cost(N)
    
    # Print the result to standard output.
    print(total_paid)