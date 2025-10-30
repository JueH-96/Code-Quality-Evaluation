class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        N = len(b)
        dp_prev = [0] * (N + 1)
        for k in range(1, 5):
            dp_curr = [float('-inf')] * (N + 1)
            for i in range(1, N + 1):
                dp_curr[i] = max(dp_curr[i - 1], dp_prev[i - 1] + a[k - 1] * b[i - 1])
            dp_prev = dp_curr
        return dp_prev[N]