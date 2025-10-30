class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        from functools import lru_cache
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()

        @lru_cache(None)
        def count(prev_idx, mask):
            if mask == (1 << n) - 1:
                return 1
            total = 0
            for i in range(n):
                if not (mask & (1 << i)):
                    if prev_idx == -1 or nums[prev_idx] % nums[i] == 0 or nums[i] % nums[prev_idx] == 0:
                        total = (total + count(i, mask | (1 << i))) % MOD
            return total

        return count(-1, 0)