from typing import List

MOD = 10**9 + 7

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        # Helper function to calculate nCr % MOD
        def nCr(n, r):
            if r > n:
                return 0
            if r == 0 or r == n:
                return 1
            if r == 1 or r == n - 1:
                return n
            num = den = 1
            for i in range(r):
                num = (num * (n - i)) % MOD
                den = (den * (i + 1)) % MOD
            return num * pow(den, MOD - 2, MOD) % MOD
        
        # Create a DP table where dp[i][j] represents the number of permutations of length i with j inversions
        dp = [[0] * (n * (n - 1) // 2 + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        # Fill the DP table
        for i in range(1, n + 1):
            for j in range(i * (i - 1) // 2 + 1):
                for k in range(min(j, i - 1) + 1):
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - k]) % MOD
        
        # Calculate the answer for each requirement
        answer = 1
        for end, cnt in requirements:
            answer = (answer * dp[end + 1][cnt]) % MOD
        
        return answer

# Example usage:
# sol = Solution()
# print(sol.numberOfPermutations(3, [[2,2],[0,0]])) # Output: 2
# print(sol.numberOfPermutations(3, [[2,2],[1,1],[0,0]])) # Output: 1
# print(sol.numberOfPermutations(2, [[0,0],[1,0]])) # Output: 1