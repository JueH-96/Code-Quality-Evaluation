from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        requirements.sort(key=lambda x: x[0])
        m = len(requirements)
        
        # dp[i][j] represents the number of permutations for the first i+1 elements
        # with j inversions
        dp = [[0]*(401) for _ in range(m)]
        
        # Initialize dp[0]
        dp[0][requirements[0][1]] = 1
        
        for i in range(1, m):
            end, cnt = requirements[i]
            prev_end, prev_cnt = requirements[i-1]
            for j in range(401):
                if dp[i-1][j] == 0:
                    continue
                for k in range(end - prev_end):
                    new_cnt = j + k
                    if new_cnt > 400:
                        break
                    dp[i][new_cnt] = (dp[i][new_cnt] + dp[i-1][j]) % MOD
        
        # The answer is the sum of dp[m-1][cnt] for all cnt
        return sum(dp[m-1]) % MOD