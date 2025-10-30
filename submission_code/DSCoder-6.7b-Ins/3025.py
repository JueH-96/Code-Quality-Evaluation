class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        nums.sort()
        target_bin = bin(target)[2:]
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = dp[i + 1] + 1
            j = i + 1
            while j < n and nums[j] < nums[i]:
                dp[i] = min(dp[i], dp[j + 1] + 1)
                j += 1
        res = float('inf')
        for i in range(n):
            if nums[i] > target:
                break
            j = i
            cur = 0
            while j < n and cur + nums[j] <= target:
                cur += nums[j]
                j += 1
            if cur == target:
                res = min(res, dp[i] + n - j)
        return res if res < float('inf') else -1