from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # Precompute the divisibility relationship
        divisible = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if nums[j] % nums[i] == 0:
                    divisible[i][j] = True
        
        # dp[mask][i] will store the number of ways to arrange the numbers
        # corresponding to the set bits in mask, ending with the i-th number
        dp = [[0] * n for _ in range(1 << n)]
        
        # Base case: single element permutations
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Fill dp table
        for mask in range(1, 1 << n):
            for i in range(n):
                # If i-th number is in the current subset (mask)
                if mask & (1 << i):
                    # Try to extend the permutation by adding a number j
                    for j in range(n):
                        # Check if j-th number can be added and is not in the current subset
                        if not mask & (1 << j) and (divisible[i][j] or divisible[j][i]):
                            dp[mask | (1 << j)][j] += dp[mask][i]
                            dp[mask | (1 << j)][j] %= MOD
        
        # Sum up all the valid permutations ending with any number
        return sum(dp[-1]) % MOD

# Example usage:
# sol = Solution()
# print(sol.specialPerm([2,3,6]))  # Output: 2
# print(sol.specialPerm([1,4,3]))  # Output: 2