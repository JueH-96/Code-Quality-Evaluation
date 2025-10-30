class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        M = 10 ** 9 + 7
        N = len(nums)
        if N < 5:
            return 0
        
        counts = [[0] * N for _ in range(1001)]
        prev = [-1] * 1001
        for i, num in enumerate(nums):
            num += 500
            counts[num][i] = 1
            if prev[num] != -1:
                for j in range(prev[num] + 1, i):
                    counts[num][i] = (counts[num][i] + counts[num][j]) % M
                counts[num][i] = (counts[num][i] * 2) % M
            prev[num] = i
            
        middle_counts = [0] * N
        for i in range(500, 1001):
            for j in range(N):
                middle_counts[j] = (middle_counts[j] + counts[i][j]) % M
        
        dp = [[0] * (N + 1) for _ in range(5)]
        for i in range(1, N):
            dp[1][i] = dp[1][i - 1] + 1
            
        for i in range(2, 5):
            for j in range(i, N):
                dp[i][j + 1] = dp[i][j] + dp[i - 1][j]
        
        ans = 0
        for i in range(4, N):
            total = sum(dp[j][-1] - dp[j][i - 3] for j in range(1, 4))
            total = (total * pow(2, M - 2, M)) % M
            ans = (ans + total * middle_counts[i]) % M
        return ans