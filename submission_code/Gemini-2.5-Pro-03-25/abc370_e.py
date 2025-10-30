# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read N and K from standard input
    # N: length of the sequence A
    # K: the forbidden sum for subsequences
    N, K = map(int, sys.stdin.readline().split())
    
    # Read sequence A from standard input
    # A is a list of integers A_1, A_2, ..., A_N
    A = list(map(int, sys.stdin.readline().split()))
    
    # Modulo constant for calculations
    MOD = 998244353

    # Problem: Count the number of ways to divide sequence A into contiguous subsequences
    # such that no subsequence sums to K. The count should be modulo MOD.

    # Approach: Dynamic Programming with Optimization
    # Let dp[i] be the number of valid ways to divide the prefix A[0...i-1].
    # We want to find dp[N].
    
    # Let P[i] be the prefix sum A[0] + ... + A[i-1]. Define P[0] = 0.
    # The sum of a contiguous subsequence A[j...i-1] (0-based indexing) is P[i] - P[j].
    
    # A division of A[0...i-1] ends with a subsequence A[j...i-1] for some 0 <= j < i.
    # This division is formed by taking a valid division of A[0...j-1] (counted by dp[j])
    # and appending the subsequence A[j...i-1].
    # This resulting division is valid if and only if the sum P[i] - P[j] is not equal to K.
    
    # The recurrence relation is:
    # dp[i] = sum(dp[j] for 0 <= j < i such that P[i] - P[j] != K)
    
    # This can be computed more efficiently. Let T[i] = sum(dp[k] for 0 <= k <= i).
    # Then T[i-1] = sum(dp[j] for 0 <= j < i).
    # The recurrence can be rewritten as:
    # dp[i] = T[i-1] - sum(dp[j] for 0 <= j < i such that P[i] - P[j] = K)
    # The condition P[i] - P[j] = K is equivalent to P[j] = P[i] - K.
    # So, dp[i] = T[i-1] - sum(dp[j] for 0 <= j < i such that P[j] = P[i] - K)
    
    # We can maintain the total sum T[i] incrementally: T[i] = T[i-1] + dp[i].
    # We use a hash map (dictionary in Python) to efficiently compute the sum of dp[j]
    # for indices j where P[j] equals a specific target value (P[i] - K).
    # The map `sum_dp_by_prefix_sum` will store: {prefix_sum_value: sum of dp[j] % MOD for all j such that P[j] = prefix_sum_value}.

    # Base case: For an empty prefix (i=0), there is one way to divide it (the empty division).
    # So, dp[0] = 1.
    # Accordingly, P[0] = 0 and T[0] = dp[0] = 1.

    # Initialize state variables
    prefix_sum = 0  # Represents P_i as we iterate. Starts with P_0 = 0.
    total_sum = 1   # Represents T[i]. Starts with T[0] = 1.
                    # At the beginning of iteration i, `total_sum` holds T[i-1].
    
    # Initialize the map with the base case information.
    # Key: prefix sum P_j. Value: Sum of dp[k] % MOD for k <= j where P_k = P_j.
    sum_dp_by_prefix_sum = {0: 1} # For P[0]=0, the sum includes dp[0]=1.

    # `last_dp_val` stores the latest computed dp value, i.e., dp[i].
    # It is initialized to dp[0] = 1. After the loop for i=N finishes, it will hold dp[N].
    last_dp_val = 1 

    # Iterate from i = 1 to N. In iteration i, we compute dp[i].
    # The loop corresponds to processing elements A[0] through A[N-1].
    for i in range(1, N + 1):
        # Update prefix sum: P_i = P_{i-1} + A[i-1]
        # A[i-1] is the i-th element of the original sequence (using 0-based index).
        prefix_sum += A[i-1] 
        
        # Calculate the target prefix sum value: P_j = P_i - K.
        # If any previous index j has P[j] equal to target_P, then the subsequence A[j...i-1] sums to K.
        target_P = prefix_sum - K
        
        # Query the map to get the sum of dp[j] for all j < i such that P[j] = target_P.
        # This sum, S_bad, represents the total count of valid divisions ending at indices j
        # which would become invalid if extended by the subsequence A[j...i-1].
        # Default to 0 if target_P has not been seen as a prefix sum before.
        S_bad = sum_dp_by_prefix_sum.get(target_P, 0)
        
        # Calculate dp[i] using the recurrence: dp[i] = (T[i-1] - S_bad) % MOD.
        # `total_sum` currently holds T[i-1].
        # We add MOD before taking the modulo to handle potential negative results from subtraction.
        current_dp = (total_sum - S_bad + MOD) % MOD
        
        # Update the total sum T[i] = (T[i-1] + dp[i]) % MOD
        # This updates `total_sum` to hold T[i] for the next iteration.
        total_sum = (total_sum + current_dp) % MOD
        
        # Update the map: Add the newly computed dp[i] (current_dp) to the sum associated
        # with the current prefix sum P_i (prefix_sum).
        # If P_i already exists in the map, add current_dp to its value. Otherwise, insert P_i with value current_dp.
        sum_dp_by_prefix_sum[prefix_sum] = (sum_dp_by_prefix_sum.get(prefix_sum, 0) + current_dp) % MOD
        
        # Store the computed dp[i] value. After the loop completes, this will hold dp[N].
        last_dp_val = current_dp

    # The final answer is dp[N], which is stored in last_dp_val.
    print(last_dp_val)

# Execute the solver function
solve()