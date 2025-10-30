class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        @lru_cache(None)
        def dfs(i: int, x: int) -> int:
            if x == 0 or i >= len(nums):
                return 0
            c = 1 if x % 2 == 1 else -1
            ans = dfs(i + 1, x)
            for j in range(i, len(nums)):
                ans = max(ans, c * sum(nums[i:j + 1]) + dfs(j + 1, x - 1))
            return ans
        return dfs(0, k)