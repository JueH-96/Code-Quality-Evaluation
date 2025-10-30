from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        max_inversions = max(req[1] for req in requirements)
        dp = [[0] * (max_inversions + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        # Sort requirements by end index
        requirements.sort(key=lambda x: x[0])
        
        for i in range(1, n + 1):
            for j in range(max_inversions + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= i:
                    dp[i][j] += dp[i][j - i]
                dp[i][j] %= MOD
        
        result = 1
        prev_end = -1
        for end, cnt in requirements:
            for j in range(cnt + 1, max_inversions + 1):
                dp[end + 1][j] = 0
            if end > prev_end:
                result *= dp[end + 1][cnt]
                result %= MOD
            prev_end = end
        
        return result