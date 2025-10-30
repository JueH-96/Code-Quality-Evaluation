import collections

class Solution:
  def countSubMultisets(self, nums: list[int], l: int, r: int) -> int:
    MOD = 10**9 + 7

    counts = collections.Counter(nums)
    
    num_zeros = counts.pop(0, 0) # Get count of 0s and remove 0 from map

    # dp[s] stores the number of ways to make sum s
    # using non-zero elements processed so far
    dp = [0] * (r + 1)
    dp[0] = 1 # Base case: one way to make sum 0 (empty multiset)

    # Iterate over distinct positive numbers (x) and their counts (c)
    for x, c in counts.items():
        # dp_for_this_x will store the dp table values after processing x.
        # It's initialized as a copy of the current dp table (dp_old).
        # This is because for sums s' not of the form rem + q*x, 
        # or for sums s = rem + q*x where the sliding window sum is based on dp_old values,
        # the value dp_old[s'] (meaning 0 occurrences of current x are chosen) is preserved.
        # The sliding window sum calculates Sum_{k=0 to c} dp_old[s - k*x].
        dp_for_this_x = dp[:] 
        
        for rem in range(x):
            current_sum_window = 0
            # Iterate q_idx from 0 upwards. Current sum s = rem + q_idx * x.
            # The window sums dp_old[rem + (q_idx-k)*x] for k from 0 to c.
            # The recurrence for current_sum_window S_q for current s (index q):
            # S_q = S_{q-1} + dp_old[s] - dp_old[s - (c+1)*x]
            # (where S_{q-1} is current_sum_window from previous q_idx for same rem)
            
            # Max q_idx such that rem + q_idx * x <= r
            max_q_idx = (r - rem) // x 
            for q_idx in range(max_q_idx + 1):
                s = rem + q_idx * x
                
                # Add dp[s] (which is dp_old[s]) to current_sum_window. This is V_q.
                current_sum_window = (current_sum_window + dp[s]) % MOD
                
                # Element dp_old[s - (c+1)*x] leaves the window. This is V_{q-c-1}.
                s_to_remove = s - (c + 1) * x
                if s_to_remove >= 0: 
                    current_sum_window = (current_sum_window - dp[s_to_remove] + MOD) % MOD
                
                # Update dp_for_this_x[s] with the calculated sum for the window ending at s
                dp_for_this_x[s] = current_sum_window
        
        dp = dp_for_this_x # Update dp table for the next distinct number

    # Calculate the sum of ways for the range [l, r]
    ans = 0
    for s_val in range(l, r + 1):
        # s_val is always within bounds of dp array [0, r] due to loop range and l,r constraints
        ans = (ans + dp[s_val]) % MOD
    
    # Account for zeros:
    # Each sub-multiset of non-zero items summing to s can be combined with
    # 0 to num_zeros items of value 0. There are (num_zeros + 1) ways to choose zeros.
    # This multiplies the count for each sum s by (num_zeros + 1).
    ans = (ans * (num_zeros + 1)) % MOD
    
    return ans