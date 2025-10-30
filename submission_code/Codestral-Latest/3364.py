class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)

        # Precompute the prefix AND for nums
        prefix_and = [0] * (n + 1)
        for i in range(n):
            prefix_and[i + 1] = prefix_and[i] & nums[i]

        # DP array to store the minimum sum of values
        dp = [float('inf')] * (m + 1)
        dp[0] = 0

        for i in range(1, m + 1):
            for j in range(i):
                if prefix_and[n] & andValues[j] == andValues[j]:
                    dp[i] = min(dp[i], dp[j] + (nums[n - 1] if i == m else nums[n - i]))

        return dp[m] if dp[m] != float('inf') else -1