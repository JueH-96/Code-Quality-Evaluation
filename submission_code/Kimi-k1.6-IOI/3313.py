class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Precompute coefficients for each subarray position (1-based)
        coeff = [0] * (k + 2)  # coeff[1] to coeff[k]
        for m in range(1, k + 1):
            sign = 1 if (m % 2 == 1) else -1
            coeff[m] = sign * (k - m + 1)
        
        # Initialize DP table
        prev_dp = [[-float('inf')] * 2 for _ in range(k + 1)]
        prev_dp[0][0] = 0
        
        for num in nums:
            curr_dp = [[-float('inf')] * 2 for _ in range(k + 1)]
            for m in range(k + 1):
                for s in [0, 1]:
                    if prev_dp[m][s] == -float('inf'):
                        continue
                    # Option 1: Do not take the current number
                    if curr_dp[m][s] < prev_dp[m][s]:
                        curr_dp[m][s] = prev_dp[m][s]
                    # Option 2: Start a new subarray (if possible)
                    if s == 0 and m < k:
                        new_m = m + 1
                        contrib = num * coeff[new_m]
                        if curr_dp[new_m][1] < prev_dp[m][s] + contrib:
                            curr_dp[new_m][1] = prev_dp[m][s] + contrib
                    # Option 3: Continue or end the current subarray
                    if s == 1:
                        # Continue the subarray
                        contrib = num * coeff[m]
                        if curr_dp[m][1] < prev_dp[m][s] + contrib:
                            curr_dp[m][1] = prev_dp[m][s] + contrib
                        # End the subarray
                        if curr_dp[m][0] < prev_dp[m][s]:
                            curr_dp[m][0] = prev_dp[m][s]
            prev_dp = curr_dp
        
        return max(prev_dp[k][0], prev_dp[k][1])