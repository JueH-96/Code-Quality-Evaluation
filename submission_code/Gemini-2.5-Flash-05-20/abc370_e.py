import sys

def solve():
    # Read N and K from the first line of input
    N, K = map(int, sys.stdin.readline().split())
    # Read the sequence A from the second line of input
    # Convert to list for 0-indexed access
    A = list(map(int, sys.stdin.readline().split()))

    # Modulo constant
    MOD = 998244353

    # current_prefix_sum: This variable stores P[i] (1-indexed prefix sum)
    # for the current iteration 'i'.
    # P[i] is the sum of A[0]...A[i-1] in 0-indexed A.
    # Initialized to 0, representing P[0] (sum of empty prefix).
    current_prefix_sum = 0 

    # total_dp_sum: This variable stores the sum of all valid ways to divide
    # prefixes A[0...j-1] for j from 0 up to the current (i) in the loop.
    # In other words, total_dp_sum = dp[0] + dp[1] + ... + dp[i] when calculating dp[i+1].
    # dp[0] (for empty prefix) is 1. So, initialize with 1.
    total_dp_sum = 1 

    # ways_at_prefix_sum: A dictionary to map prefix sum values to the sum of
    # dp values that resulted in that prefix sum.
    # Specifically, ways_at_prefix_sum[val] = sum(dp[j]) for all j such that P[j] == val.
    # Initialized with P[0] = 0, dp[0] = 1.
    ways_at_prefix_sum = {0: 1} 

    # dp_N_val: This variable will ultimately hold the final answer, dp[N].
    # It gets updated in each iteration to be dp[i+1].
    dp_N_val = 0 

    # Iterate through the array A from index 0 to N-1.
    # In each iteration 'i', we effectively calculate dp[i+1], which is the number of valid
    # divisions for the prefix A[0...i].
    for i in range(N):
        # Update current_prefix_sum by adding A[i].
        # Now current_prefix_sum represents P[i+1] (sum of A[0]...A[i]).
        current_prefix_sum += A[i] 

        # Calculate 'bad_ways_count': These are divisions of A[0...i] that end
        # with a segment A[j...i] whose sum is K.
        # The sum A[j...i] = P[i+1] - P[j].
        # If P[i+1] - P[j] == K, then P[j] must be P[i+1] - K.
        # We need to find the sum of dp[j] for all such j (where j is 0-indexed for dp).
        # This is exactly what ways_at_prefix_sum.get(P[i+1] - K, 0) provides.
        bad_ways_count = ways_at_prefix_sum.get(current_prefix_sum - K, 0)

        # Calculate dp_val_for_current_segment (which is dp[i+1]):
        # It's the total ways to divide A[0...i] (represented by total_dp_sum,
        # which is sum(dp[0]...dp[i]))
        # minus the 'bad ways' (those ending with a segment summing to K).
        # We add MOD before taking modulo to handle potential negative results from subtraction.
        dp_val_for_current_segment = (total_dp_sum - bad_ways_count + MOD) % MOD

        # Update total_dp_sum for the next iteration.
        # It now includes the newly calculated dp_val_for_current_segment (dp[i+1]).
        total_dp_sum = (total_dp_sum + dp_val_for_current_segment) % MOD

        # Update the ways_at_prefix_sum dictionary.
        # Add the current dp_val_for_current_segment (dp[i+1]) to the sum
        # associated with its corresponding prefix sum (current_prefix_sum, which is P[i+1]).
        ways_at_prefix_sum[current_prefix_sum] = (ways_at_prefix_sum.get(current_prefix_sum, 0) + dp_val_for_current_segment) % MOD
        
        # Store the current dp value. After the loop completes, this will be dp[N].
        dp_N_val = dp_val_for_current_segment

    # Print the final result (dp[N])
    sys.stdout.write(str(dp_N_val) + "
")

# Call the solve function to execute the program
solve()