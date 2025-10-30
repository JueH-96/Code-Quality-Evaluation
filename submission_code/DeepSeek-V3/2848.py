class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        from functools import lru_cache
        
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute the adjacency list where adj[i] contains all j such that nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0):
                    adj[i].append(j)
        
        # dp[mask][last] represents the number of ways to form a permutation with the elements in mask, ending with last
        @lru_cache(maxsize=None)
        def dp(mask, last):
            if mask == (1 << n) - 1:
                return 1
            total = 0
            for next_num in adj[last]:
                if not (mask & (1 << next_num)):
                    total += dp(mask | (1 << next_num), next_num)
                    total %= MOD
            return total
        
        result = 0
        for i in range(n):
            result += dp(1 << i, i)
            result %= MOD
        return result