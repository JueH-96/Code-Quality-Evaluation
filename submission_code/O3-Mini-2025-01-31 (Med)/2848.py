class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute valid adjacent relations
        can_follow = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                        can_follow[i][j] = True

        # dp[mask][i] = number of ways to reach state (mask) ending with element at index i.
        dp = [[0]*n for _ in range(1 << n)]
        # Base case: each number can be the start of the permutation
        for i in range(n):
            dp[1 << i][i] = 1
            
        for mask in range(1 << n):
            for i in range(n):
                if dp[mask][i]:
                    for j in range(n):
                        # j is not in mask and must be adjacent valid
                        if (mask >> j) & 1 == 0 and can_follow[i][j]:
                            new_mask = mask | (1 << j)
                            dp[new_mask][j] = (dp[new_mask][j] + dp[mask][i]) % MOD

        full_mask = (1 << n) - 1
        result = 0
        for i in range(n):
            result = (result + dp[full_mask][i]) % MOD
        return result