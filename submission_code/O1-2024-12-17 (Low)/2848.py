class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        import functools
        
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute whether two elements can follow each other
        # i.e., nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                        adj[i][j] = True
        
        @functools.lru_cache(None)
        def dp(mask, last):
            # If we have used all elements, we have formed a valid permutation
            if mask == (1 << n) - 1:
                return 1
            ans = 0
            for nxt in range(n):
                if not (mask & (1 << nxt)):  # if nxt is not used yet
                    # We can only use nxt if it forms a valid adjacent pair with last
                    if last is None or adj[last][nxt]:
                        ans += dp(mask | (1 << nxt), nxt)
            return ans % MOD
        
        # We can start from any element as the first in permutation (no 'last' constraint)
        return dp(0, None)