class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        from collections import Counter
        import heapq
        
        n = len(nums)
        dp = [0] + [-1e9] * k
        dp2 = [0] + [-1e9] * k
        for x in nums:
            dp2 = [min(dp[j] + (x != j) for j in range(k))] + [1e9] * k
            dp = dp2[:-1]
        return dp[nums[0]]