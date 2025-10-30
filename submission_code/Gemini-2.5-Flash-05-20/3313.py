import math

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)

        # Precompute prefix sums
        # P[x] stores sum(nums[0...x-1])
        # sum(nums[a...b]) = P[b+1] - P[a]
        P = [0] * (n + 1)
        for i in range(n):
            P[i+1] = P[i] + nums[i]

        # dp_prev represents dp[j-1], dp_current represents dp[j]
        # dp[j][i] = max strength using j subarrays, with the j-th subarray ending at index i.
        # Initialize with -infinity, as strengths can be negative.
        dp_current = [-math.inf] * n
        dp_prev = [-math.inf] * n

        # Iterate for each number of subarrays from 1 to k
        for j_count in range(1, k + 1):
            # Calculate the coefficient for the j_count-th subarray
            # C_j = (-1)**(j_count + 1) * (k - j_count + 1)
            C_j = (1 if (j_count + 1) % 2 == 0 else -1) * (k - j_count + 1)
            
            if j_count == 1:
                # Special handling for j_count = 1
                # dp[1][i] = max_{0 <= p <= i} (P[i+1] - P[p]) * C_1
                # Since C_1 = k is positive (k is positive odd integer),
                # this is (P[i+1] - min_{0 <= p <= i} P[p]) * C_1
                min_P_val = math.inf # min_P_val stores min(P[0], ..., P[i])
                for i in range(n):
                    min_P_val = min(min_P_val, P[i])
                    dp_current[i] = (P[i+1] - min_P_val) * C_j
            else:
                # For j_count > 1, use the optimized DP transition
                # dp[j][i] = P[i+1] * C_j + max_{0 <= p <= i} ( -P[p] * C_j + dp[j-1][p-1] )
                # Let f(p) = -P[p] * C_j + dp[j-1][p-1]
                # max_f_val stores max_{0 <= p <= i} f(p)
                max_f_val = -math.inf 
                
                for i in range(n):
                    # p is the potential starting index of the j_count-th subarray.
                    # The previous (j_count-1) subarrays must end at p-1.
                    # This means we need dp_prev[p-1].
                    # The earliest index (p-1) for (j_count-1) subarrays is (j_count-1)-1 = j_count-2.
                    # So, if i-1 < j_count-2, dp_prev[i-1] is not a valid end for j_count-1 subarrays.
                    
                    prev_dp_val_for_f_i = -math.inf
                    # Check if i-1 is a valid end index for j_count-1 subarrays
                    if i - 1 >= j_count - 2: 
                        # Check if dp_prev[i-1] itself is a reachable state (not -inf)
                        if dp_prev[i-1] != -math.inf: 
                            prev_dp_val_for_f_i = dp_prev[i-1]
                    
                    current_f_i = -math.inf
                    if prev_dp_val_for_f_i != -math.inf:
                         current_f_i = -P[i] * C_j + prev_dp_val_for_f_i
                    
                    # Update max_f_val (running maximum of f(p) for p up to current i)
                    if current_f_i != -math.inf:
                        max_f_val = max(max_f_val, current_f_i)

                    # Calculate dp_current[i] using the accumulated max_f_val
                    if max_f_val != -math.inf:
                        dp_current[i] = P[i+1] * C_j + max_f_val
                    else:
                        dp_current[i] = -math.inf # Not possible to form j_count subarrays ending at i
            
            # After filling dp_current for j_count, copy it to dp_prev for the next iteration
            dp_prev = list(dp_current)
            # Reset dp_current for the next j_count iteration (not strictly necessary but good practice)
            dp_current = [-math.inf] * n

        # The final answer is the maximum value in dp_prev (which now holds dp[k])
        # We need to find the max strength over all possible ending indices for the k-th subarray.
        max_strength = -math.inf
        for val in dp_prev:
            max_strength = max(max_strength, val)
        
        return max_strength