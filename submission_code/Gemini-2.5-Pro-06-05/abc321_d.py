import sys
from bisect import bisect_left

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Read problem parameters from standard input
    try:
        N, M, P = map(int, input().split())
        A = list(map(int, input().split()))
        B = list(map(int, input().split()))
    except (ValueError, IndexError):
        # Handles cases where input might be empty
        return

    # Sort the list of side dish prices for efficient searching.
    B.sort()

    # Precompute the prefix sums of the sorted side dish prices.
    # prefix_B[k] will store the sum of the first k elements of B.
    # This allows for O(1) lookup of sums of initial segments.
    prefix_B = [0] * (M + 1)
    for i in range(M):
        prefix_B[i + 1] = prefix_B[i] + B[i]

    # Initialize the total price of all set meals.
    total_price = 0

    # Iterate through each main dish.
    for a in A:
        # For a main dish 'a', we partition side dishes 'b' based on whether a + b < P.
        # This is equivalent to b < P - a.
        threshold = P - a
        
        # Use binary search (bisect_left) to find the number of side dishes 'b'
        # with price strictly less than the threshold. Let this be 'k'.
        k = bisect_left(B, threshold)
        
        # --- Calculate sum for the 'k' pairs where price is a + b ---
        # The sum is sum(a + b_j for j=0..k-1) = k*a + (B[0] + ... + B[k-1]).
        # The sum of B's is retrieved from our prefix sum array.
        sum_for_cheaper_pairs = k * a + prefix_B[k]
        
        # --- Calculate sum for the 'M - k' pairs where price is P ---
        # For the remaining pairs, the price is capped at P.
        sum_for_capped_pairs = (M - k) * P
        
        # Add the contributions for the current main dish 'a' to the total.
        total_price += sum_for_cheaper_pairs + sum_for_capped_pairs
        
    # Print the final total price.
    print(total_price)

# The problem is solved by executing the function.
solve()