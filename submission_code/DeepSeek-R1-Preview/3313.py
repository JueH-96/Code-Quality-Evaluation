class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        # Precompute the weights for each subarray position
        weight = [0] * (k + 1)
        for j in range(1, k + 1):
            sign = (-1) ** (j + 1)
            multiplier = (k - j + 1)
            weight[j] = sign * multiplier
        
        # Initialize the DP arrays
        prev_dp = [[-float('inf')] * 2 for _ in range(k + 1)]
        prev_dp[0][0] = 0  # Starting state: 0 subarrays, not in a subarray
        
        for num in nums:
            curr_dp = [[-float('inf')] * 2 for _ in range(k + 1)]
            for j in range(k + 1):
                for s in [0, 1]:
                    current_val = prev_dp[j][s]
                    if current_val == -float('inf'):
                        continue
                    if s == 0:
                        # Option 1: Stay in state 0
                        if curr_dp[j][0] < current_val:
                            curr_dp[j][0] = current_val
                        # Option 2: Start a new subarray if possible
                        if j < k:
                            new_val = current_val + num * weight[j + 1]
                            if new_val > curr_dp[j][1]:
                                curr_dp[j][1] = new_val
                    else:
                        # Option 1: Continue the current subarray
                        new_val = current_val + num * weight[j + 1]
                        if new_val > curr_dp[j][1]:
                            curr_dp[j][1] = new_val
                        # Option 2: End the current subarray
                        if j + 1 <= k:
                            new_val = current_val + num * weight[j + 1]
                            if new_val > curr_dp[j + 1][0]:
                                curr_dp[j + 1][0] = new_val
            # Update previous DP for the next iteration
            prev_dp = curr_dp
        
        # The answer is the maximum strength after selecting exactly k subarrays and ending in state 0
        return prev_dp[k][0]