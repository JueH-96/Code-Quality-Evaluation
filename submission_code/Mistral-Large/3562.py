from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        intervals = sorted(enumerate(intervals), key=lambda x: (x[1][0], x[1][1], -x[1][2]))

        dp = [[-1] * 5 for _ in range(n)]
        dp[0][1] = intervals[0][0]
        max_score = [0] * n
        max_score[0] = intervals[0][1][2]

        for i in range(1, n):
            for j in range(1, 5):
                dp[i][j] = dp[i-1][j]
                max_score[i] = max_score[i-1]

            for j in range(1, 5):
                k = i
                while k > 0 and intervals[k][1][0] <= intervals[i][1][1]:
                    k -= 1
                if k == -1 or intervals[k][1][1] < intervals[i][1][0]:
                    if dp[k][j-1] != -1 and (dp[i][j] == -1 or max_score[k] + intervals[i][1][2] > max_score[i]):
                        dp[i][j] = dp[k][j-1] + [intervals[i][0]]
                        max_score[i] = max_score[k] + intervals[i][1][2]

        best_indices = dp[n-1][4] if dp[n-1][4] != -1 else dp[n-1][3] if dp[n-1][3] != -1 else dp[n-1][2] if dp[n-1][2] != -1 else dp[n-1][1]
        return best_indices