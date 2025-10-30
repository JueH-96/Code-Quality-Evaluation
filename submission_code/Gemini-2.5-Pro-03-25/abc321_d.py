import sys
import bisect

# Function to solve the problem
def solve():
    # Read N, M, P from stdin
    # N: number of main dishes
    # M: number of side dishes
    # P: the price cap for a set meal
    N, M, P = map(int, sys.stdin.readline().split())
    
    # Read list A (prices of main dishes) from stdin
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read list B (prices of side dishes) from stdin
    B = list(map(int, sys.stdin.readline().split()))

    # Sort the list B of side dish prices in non-decreasing order.
    # This allows us to efficiently find how many side dishes satisfy a condition using binary search,
    # and to use prefix sums for quick calculation of sums over ranges.
    B.sort()

    # Compute prefix sums for the sorted list B.
    # prefix_sum_B[k] will store the sum B[0] + B[1] + ... + B[k-1].
    # The array size is M+1 to handle sums up to the full list B.
    # prefix_sum_B[0] is initialized to 0.
    prefix_sum_B = [0] * (M + 1)
    for k in range(M):
        prefix_sum_B[k+1] = prefix_sum_B[k] + B[k]

    # Initialize the total price accumulator. This will store the sum of prices of all NM possible set meals.
    total_price = 0

    # Iterate through each main dish price A_i in list A.
    for i in range(N):
        current_A = A[i]  # The price of the current main dish being considered.
        
        # Calculate the threshold value for a side dish price B_j such that the sum A_i + B_j is less than P.
        # The condition is A_i + B_j < P, which is equivalent to B_j < P - A_i.
        # We call this threshold 'target'.
        target = P - current_A
        
        # Use binary search (bisect_left from the bisect module) to find the insertion point `idx` for `target` in the sorted list `B`.
        # `bisect_left(B, target)` returns the smallest index `idx` such that B[idx] >= target.
        # This means all elements B[k] for k < idx satisfy B[k] < target.
        # The number of such elements (side dishes B_j with price < P - A_i) is exactly `idx`.
        idx = bisect.bisect_left(B, target)
        
        # `count_less_than_P` is the number of side dishes B_j such that A_i + B_j < P. This equals `idx`.
        count_less_than_P = idx
        
        # `sum_B_less_than_P` is the sum of prices of these side dishes (B[0] + ... + B[idx-1]).
        # We can retrieve this sum efficiently using the precomputed prefix sums: prefix_sum_B[idx].
        sum_B_less_than_P = prefix_sum_B[idx]
        
        # `count_geq_P` is the number of side dishes B_j such that A_i + B_j >= P.
        # This is the total number of side dishes M minus the count of those where the sum is less than P.
        count_geq_P = M - idx
        
        # Calculate the total price contribution for set meals involving the current main dish A_i.
        # For the `count_less_than_P` side dishes where A_i + B_j < P, the set meal price is A_i + B_j.
        # The sum of these prices is sum(A_i + B_j) = sum(A_i) + sum(B_j) = count_less_than_P * A_i + sum_B_less_than_P.
        # For the `count_geq_P` side dishes where A_i + B_j >= P, the set meal price is P.
        # The sum of these prices is sum(P) = count_geq_P * P.
        # The total contribution for A_i is the sum of these two parts.
        current_A_total = (count_less_than_P * current_A + sum_B_less_than_P) + (count_geq_P * P)
        
        # Add this contribution to the overall total price.
        total_price += current_A_total

    # Print the final calculated total price to standard output.
    print(total_price)

# Call the solve function to execute the logic based on the input read from stdin.
solve()