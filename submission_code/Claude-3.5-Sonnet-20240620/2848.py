class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        @functools.lru_cache(None)
        def dfs(mask, prev):
            if mask == (1 << n) - 1:
                return 1
            
            count = 0
            for i in range(n):
                if mask & (1 << i):
                    continue
                if prev == -1 or nums[i] % nums[prev] == 0 or nums[prev] % nums[i] == 0:
                    count += dfs(mask | (1 << i), i)
            
            return count % MOD
        
        return dfs(0, -1)