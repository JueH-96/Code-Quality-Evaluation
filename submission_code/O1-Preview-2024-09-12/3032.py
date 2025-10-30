class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        MAX_LOG = 35  # Since k can be up to 1e10, log2(1e10) ≈ 34

        dp = [[0]*MAX_LOG for _ in range(n)]
        sum_dp = [[0]*MAX_LOG for _ in range(n)]

        # Initialize dp and sum_dp for i=0
        for x in range(n):
            dp[x][0] = receiver[x]
            sum_dp[x][0] = receiver[x]

        # Precompute dp and sum_dp for all logs
        for i in range(1, MAX_LOG):
            for x in range(n):
                dp_x_i_minus_1 = dp[x][i-1]
                dp[x][i] = dp[dp_x_i_minus_1][i-1]
                sum_dp[x][i] = sum_dp[x][i-1] + sum_dp[dp_x_i_minus_1][i-1]

        max_total = 0
        for x in range(n):
            total = x
            current_x = x
            current_k = k
            for i in range(MAX_LOG-1, -1, -1):
                if current_k >= (1 << i):
                    total += sum_dp[current_x][i]
                    current_x = dp[current_x][i]
                    current_k -= (1 << i)
            max_total = max(max_total, total)
        return max_total