class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7

        # Dictionary to store the last occurrence of each number
        last_occurrence = {}
        for i, num in enumerate(nums):
            last_occurrence[num] = i

        # Dynamic programming array to store the number of good partitions
        dp = [0] * (len(nums) + 1)
        dp[0] = 1

        # Variable to keep track of the maximum end index of the current partition
        max_end = 0

        for i in range(1, len(nums) + 1):
            max_end = max(max_end, last_occurrence[nums[i - 1]])
            if max_end == i - 1:
                dp[i] = (dp[i - 1] * 2) % MOD
            else:
                dp[i] = dp[i - 1]

        return dp[len(nums)]