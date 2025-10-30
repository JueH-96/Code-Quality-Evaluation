class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        can_follow = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and (nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0):
                    can_follow[i][j] = True
        
        dp = [[0] * n for _ in range(1 << n)]
        for i in range(n):
            dp[1 << i][i] = 1
        
        for mask in range(1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                current_count = dp[mask][last]
                if current_count == 0:
                    continue
                for next_idx in range(n):
                    if not (mask & (1 << next_idx)) and can_follow[last][next_idx]:
                        new_mask = mask | (1 << next_idx)
                        dp[new_mask][next_idx] = (dp[new_mask][next_idx] + current_count) % MOD
        
        return sum(dp[(1 << n) - 1][last] for last in range(n)) % MOD