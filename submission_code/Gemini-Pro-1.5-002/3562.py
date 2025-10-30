class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        for i in range(n):
            intervals[i].append(i)
        intervals.sort()

        dp = [[(0, []) for _ in range(5)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            l, r, w, idx = intervals[i - 1]
            for j in range(5):
                dp[i][j] = dp[i - 1][j]

            for j in range(1, 5):
                prev_i = -1
                for k in range(i - 1, -1, -1):
                    if intervals[k][1] < l:
                        prev_i = k
                        break
                
                if prev_i != -1:
                    prev_score, prev_indices = dp[prev_i + 1][j - 1]
                    if prev_score + w > dp[i][j][0]:
                        dp[i][j] = (prev_score + w, sorted(prev_indices + [idx]))
                    elif prev_score + w == dp[i][j][0]:
                        new_indices = sorted(prev_indices + [idx])
                        if new_indices < dp[i][j][1]:
                            dp[i][j] = (prev_score + w, new_indices)
                else:
                    if w > dp[i][j][0]:
                        dp[i][j] = (w, [idx])
                    elif w == dp[i][j][0] and [idx] < dp[i][j][1]:
                        dp[i][j] = (w, [idx])
        
        best_score = 0
        best_indices = []
        for j in range(5):
            if dp[n][j][0] > best_score:
                best_score = dp[n][j][0]
                best_indices = dp[n][j][1]
            elif dp[n][j][0] == best_score and dp[n][j][1] < best_indices:
                best_indices = dp[n][j][1]

        return best_indices