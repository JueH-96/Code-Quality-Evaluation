# YOUR CODE HERE
import sys

def solve():
    """
    Solves the problem by reformulating it as a 0/1 knapsack problem with two constraints.
    The goal is to find the maximum number of dishes that can be chosen such that their
    total sweetness and saltiness do not exceed the limits, and then add one more dish.
    """
    # Fast I/O
    try:
        input = sys.stdin.readline
    except (IOError, NameError):
        pass

    # Read inputs
    N, X, Y = map(int, input().split())
    dishes = [tuple(map(int, input().split())) for _ in range(N)]

    # dp[j][sa] = minimum total saltiness to select j dishes with total sweetness sa.
    # Initialize with a value larger than any possible saltiness sum.
    inf = float('inf')
    dp = [[inf] * (X + 1) for _ in range(N + 1)]

    # Base case: 0 dishes, 0 sweetness, 0 saltiness.
    dp[0][0] = 0

    dishes_considered = 0
    # Iterate through each dish to build up the DP table
    for a, b in dishes:
        dishes_considered += 1
        # Iterate downwards to ensure each dish is used at most once per state calculation
        # This is a standard technique for space-optimized 0/1 knapsack DP.
        for j in range(dishes_considered, 0, -1):
            for sa in range(X, a - 1, -1):
                # If a valid state exists for j-1 dishes with sa-a sweetness,
                # we can transition to a state with j dishes by adding the current dish.
                if dp[j - 1][sa - a] != inf:
                    new_sb = dp[j - 1][sa - a] + b
                    dp[j][sa] = min(dp[j][sa], new_sb)

    # Find j_max: the maximum number of dishes that can be eaten without violation.
    # This means finding the max j such that for some sa <= X, dp[j][sa] <= Y.
    j_max = 0
    # Search from the largest possible j downwards for efficiency.
    for j in range(N, -1, -1):
        # For a given number of dishes j, check if there's any sweetness `sa`
        # for which the total saltiness is within the limit Y.
        for sa in range(X + 1):
            if dp[j][sa] <= Y:
                j_max = j
                # Found the largest j, so we can break out of both loops.
                break
        if j_max == j:
            break

    # The maximum number of dishes Snuke eats is j_max + 1 (for the final dish
    # that might cause a violation), but it cannot exceed the total number of dishes N.
    ans = min(N, j_max + 1)
    print(ans)

solve()