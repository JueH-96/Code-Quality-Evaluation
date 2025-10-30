class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        @lru_cache(None)
        def dp(i: int, j: int, k: int) -> bool:
            if i == j:
                return True
            if k == 1:
                return prefix_sum[j] - prefix_sum[i] >= m
            for p in range(i, j):
                if prefix_sum[p] - prefix_sum[i] >= m and dp(i, p, 1) and dp(p, j, k - 1):
                    return True
            return False

        return dp(0, n, n)