# YOUR CODE HERE
memo = {}

def solve(n):
    """
    Calculates the total cost to break down integer n until all parts are less than 2.
    Uses memoization to store results of subproblems.
    """
    # Base case: If n is less than 2 (i.e., 0 or 1), no operations can be performed on it,
    # so the cost is 0.
    if n < 2:
        return 0
    
    # Memoization: If the result for n is already computed, return it.
    if n in memo:
        return memo[n]

    # Calculate the two integers to write on the blackboard: floor(n/2) and ceil(n/2).
    # In Python integer arithmetic:
    #   n // 2 gives floor(n/2).
    #   n - (n // 2) gives ceil(n/2).
    part1 = n // 2  # floor(n/2)
    part2 = n - part1 # ceil(n/2)
    
    # The cost for processing n is n itself, plus the sum of costs for processing part1 and part2.
    total_cost = n + solve(part1) + solve(part2)
    
    # Store the result in the memoization table before returning.
    memo[n] = total_cost
    return total_cost

if __name__ == '__main__':
    N = int(input())
    result = solve(N)
    print(result)