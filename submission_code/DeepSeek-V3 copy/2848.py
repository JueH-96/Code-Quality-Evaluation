class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        from functools import lru_cache
        
        n = len(nums)
        MOD = 10**9 + 7
        
        # Precompute the adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0):
                    adj[i].append(j)
        
        # Use dynamic programming with memoization
        @lru_cache(maxsize=None)
        def dp(mask, last):
            if mask == (1 << n) - 1:
                return 1
            total = 0
            for i in range(n):
                if not (mask & (1 << i)) and i in adj[last]:
                    total += dp(mask | (1 << i), i)
                    total %= MOD
            return total
        
        result = 0
        for i in range(n):
            result += dp(1 << i, i)
            result %= MOD
        return result