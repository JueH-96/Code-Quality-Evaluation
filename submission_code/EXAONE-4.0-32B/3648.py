class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        NEG_INF = -10**18
        dp = [[NEG_INF] * n for _ in range(n)]
        dp[n-1][n-1] = fruits[0][0] + fruits[0][n-1] + fruits[n-1][0]
        
        for i in range(0, n-1):
            new_dp = [[NEG_INF] * n for _ in range(n)]
            low_bound = max(0, n-1 - i)
            high_bound = min(n-1, n-1 + i)
            for j in range(low_bound, high_bound + 1):
                for k in range(low_bound, high_bound + 1):
                    if dp[j][k] == NEG_INF:
                        continue
                    for dj in (-1, 0, 1):
                        nj = j + dj
                        if nj < 0 or nj >= n:
                            continue
                        for dk in (-1, 0, 1):
                            nk = k + dk
                            if nk < 0 or nk >= n:
                                continue
                            bonus = fruits[i+1][i+1]
                            if i+1 != nj:
                                bonus += fruits[i+1][nj]
                            if i+1 != nk:
                                bonus += fruits[nk][i+1]
                            total_val = dp[j][k] + bonus
                            if total_val > new_dp[nj][nk]:
                                new_dp[nj][nk] = total_val
            dp = new_dp
        
        return dp[n-1][n-1]