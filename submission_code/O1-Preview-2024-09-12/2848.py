class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        from functools import lru_cache
        MOD = 10 ** 9 + 7
        n = len(nums)
        full_mask = (1 << n) -1
        
        @lru_cache(None)
        def dp(mask, last_idx):
            if mask == full_mask:
                return 1
            total_ways = 0
            for next_idx in range(n):
                if not mask & (1 << next_idx):
                    if nums[last_idx] % nums[next_idx] == 0 or nums[next_idx] % nums[last_idx] ==0:
                        total_ways += dp(mask | (1 << next_idx), next_idx)
                        total_ways %= MOD  # To prevent overflow
            return total_ways
            
        total = 0
        for start_idx in range(n):
            total += dp(1 << start_idx, start_idx)
            total %= MOD
        return total