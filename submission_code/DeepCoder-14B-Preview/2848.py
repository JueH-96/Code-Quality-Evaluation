MOD = 10**9 + 7

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        max_mask = 1 << n
        dp = [[0] * n for _ in range(max_mask)]
        
        # Initialize the DP table
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Iterate over all possible masks
        for mask in range(max_mask):
            for last in range(n):
                if not (mask & (1 << last)) or dp[mask][last] == 0:
                    continue
                # Try adding each possible next element
                for next in range(n):
                    if (mask & (1 << next)) == 0:
                        # Check if the last element and next can be adjacent
                        if (nums[last] % nums[next] == 0) or (nums[next] % nums[last] == 0):
                            new_mask = mask | (1 << next)
                            dp[new_mask][next] = (dp[new_mask][next] + dp[mask][last]) % MOD
        
        full_mask = (1 << n) - 1
        total = sum(dp[full_mask]) % MOD
        return total