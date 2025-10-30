import sys
import bisect

def solve():
    # Read N, M, P from the first line
    N, M, P = map(int, sys.stdin.readline().split())

    # Read array A
    A = list(map(int, sys.stdin.readline().split()))

    # Read array B
    B = list(map(int, sys.stdin.readline().split()))

    # Sort array B to enable efficient searching (binary search)
    B.sort()

    # Precompute prefix sums for array B.
    # pref_B[k] will store the sum of the first k elements of B.
    # pref_B[0] = 0, pref_B[1] = B[0], pref_B[2] = B[0] + B[1], etc.
    pref_B = [0] * (M + 1)
    for i in range(M):
        pref_B[i+1] = pref_B[i] + B[i]

    total_price = 0

    # Iterate through each main dish A_i
    for a_i in A:
        # Determine the threshold for B_j:
        # If A_i + B_j < P, price is A_i + B_j. This means B_j < P - A_i.
        # If A_i + B_j >= P, price is P. This means B_j >= P - A_i.
        
        # The target value for binary search in B
        target_B_value = P - a_i

        # Find the insertion point for target_B_value in B.
        # 'idx' is the count of elements in B that are less than target_B_value.
        # These are B[0], B[1], ..., B[idx-1].
        idx = bisect.bisect_left(B, target_B_value)

        # --- Part 1: Combinations where A_i + B_j < P ---
        # These are the first 'idx' elements of B (B[0] to B[idx-1]).
        count_less = idx
        # Sum of these B_j values using prefix sums
        sum_less_Bs = pref_B[idx] 
        
        # Contribution for these combinations: (A_i * count_less) + sum(B_j for B_j < target_B_value)
        contribution_less = (a_i * count_less) + sum_less_Bs

        # --- Part 2: Combinations where A_i + B_j >= P ---
        # These are the remaining elements of B (B[idx] to B[M-1]).
        count_ge = M - idx
        
        # Contribution for these combinations: P * count_ge
        contribution_ge = P * count_ge

        # Add contributions for the current A_i to the total price
        total_price += contribution_less + contribution_ge
    
    # Print the final total price
    sys.stdout.write(str(total_price) + '
')

solve()