import sys

def solve():
    N = int(sys.stdin.readline())
    
    # A_values will be an iterator over the integer changes A_i
    A_values = map(int, sys.stdin.readline().split())

    current_prefix_sum = 0  # Stores S_k, the cumulative sum of changes up to stop k. Initially S_0 = 0.
    
    # max_of_neg_prefix_sums stores max(-S_j) for 0 <= j <= k.
    # It represents the minimum number of passengers P_initial needed to ensure
    # that passenger count never drops below zero up to stop k.
    # Initialized with -S_0 = 0.
    max_of_neg_prefix_sums = 0 
    
    for change_val in A_values: # Loop runs N times
        current_prefix_sum += change_val # Update S_k based on current A_i
        
        # Update the requirement for P_initial.
        # P_initial must be at least -current_prefix_sum.
        # So, P_initial_min is the max of all such -S_k encountered.
        max_of_neg_prefix_sums = max(max_of_neg_prefix_sums, -current_prefix_sum)
    
    # After the loop:
    # current_prefix_sum is S_N (the total net change after N stops).
    # max_of_neg_prefix_sums is P_initial_min = max_{0 <= k <= N}(-S_k).
    
    # The minimum possible current number of passengers is P_initial_min + S_N.
    result = max_of_neg_prefix_sums + current_prefix_sum
    
    sys.stdout.write(str(result) + "
")

if __name__ == '__main__':
    solve()