from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Initialize a 2D array to store the number of subsequences with sum equal to j
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        # Base case: there is one subsequence with sum 0 (the empty subsequence)
        for i in range(n + 1):
            dp[i][0] = 1
        
        # Fill up the dp array
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                # If the current number is greater than the target sum, skip it
                if nums[i - 1] > j:
                    dp[i][j] = dp[i - 1][j]
                else:
                    # The number of subsequences with sum j is the sum of the number of subsequences with sum j
                    # without the current number and the number of subsequences with sum j - nums[i - 1]
                    dp[i][j] = (dp[i - 1][j] + dp[i - 1][j - nums[i - 1]]) % MOD
        
        # The power of the array is the number of subsequences with sum equal to k
        power = dp[n][k]
        
        # Calculate the sum of power of all subsequences
        total_power = 0
        for mask in range(1, 1 << n):
            subsequence = [nums[i] for i in range(n) if (mask & (1 << i))]
            subsequence_power = 0
            # Calculate the power of the subsequence
            subsequence_dp = [[0] * (k + 1) for _ in range(len(subsequence) + 1)]
            for i in range(len(subsequence) + 1):
                subsequence_dp[i][0] = 1
            for i in range(1, len(subsequence) + 1):
                for j in range(1, k + 1):
                    if subsequence[i - 1] > j:
                        subsequence_dp[i][j] = subsequence_dp[i - 1][j]
                    else:
                        subsequence_dp[i][j] = (subsequence_dp[i - 1][j] + subsequence_dp[i - 1][j - subsequence[i - 1]]) % MOD
            subsequence_power = subsequence_dp[len(subsequence)][k]
            total_power = (total_power + subsequence_power) % MOD
        
        return total_power