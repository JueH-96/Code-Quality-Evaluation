class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        forward_dp = [1] * n
        backward_dp = [1] * n

        for i in range(n):
            max_prev_len = 0
            for j in range(n):
                if i != j and coordinates[j][0] < coordinates[i][0] and coordinates[j][1] < coordinates[i][1]:
                    max_prev_len = max(max_prev_len, forward_dp[j])
            forward_dp[i] = 1 + max_prev_len

        for i in range(n):
            max_next_len = 0
            for j in range(n):
                if i != j and coordinates[i][0] < coordinates[j][0] and coordinates[i][1] < coordinates[j][1]:
                    max_next_len = max(max_next_len, backward_dp[j])
            backward_dp[i] = 1 + max_next_len

        return forward_dp[k] + backward_dp[k] - 1