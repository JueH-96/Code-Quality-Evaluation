import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # current_sum tracks the cumulative sum of A_i up to the current stop (S_k)
    current_sum = 0
    
    # min_prefix_sum_so_far tracks the minimum value of current_sum encountered so far,
    # including the implicit S_0 = 0.
    # This is initialized to 0 because P_0 must also satisfy P_0 + S_0 >= 0,
    # which implies P_0 >= 0, and S_0 is effectively 0.
    min_prefix_sum_so_far = 0 

    # Iterate through the changes in passengers at each stop
    for change in A:
        current_sum += change
        # Update the minimum prefix sum encountered.
        # This minimum value (or its negative) determines the minimum P_0 needed.
        min_prefix_sum_so_far = min(min_prefix_sum_so_far, current_sum)

    # The minimum initial number of passengers (P_0_min) required
    # is the amount needed to ensure that no intermediate passenger count
    # (P_0 + S_k) drops below zero.
    # This is equivalent to P_0_min = -min(S_0, S_1, ..., S_N).
    min_initial_passengers = -min_prefix_sum_so_far

    # The final current number of passengers is P_N = P_0 + S_N.
    # We want the minimum P_N, so we use P_0_min and the final S_N (which is current_sum).
    min_current_passengers = min_initial_passengers + current_sum

    print(min_current_passengers)

solve()