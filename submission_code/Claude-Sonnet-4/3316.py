class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        
        from functools import lru_cache
        
        INF = 2 * 10**8  # Large enough value
        
        @lru_cache(None)
        def dp(pos, selected, last_idx, min_diff):
            if selected == k:
                return min_diff if min_diff < INF else 0
            
            if pos == n or selected + (n - pos) < k:
                return 0
            
            # Skip current element
            result = dp(pos + 1, selected, last_idx, min_diff)
            
            # Take current element
            new_min_diff = min_diff
            if last_idx != -1:
                diff = nums[pos] - nums[last_idx]  # Since sorted, always positive
                new_min_diff = min(min_diff, diff)
            
            result += dp(pos + 1, selected + 1, pos, new_min_diff)
            
            return result % MOD
        
        return dp(0, 0, -1, INF)