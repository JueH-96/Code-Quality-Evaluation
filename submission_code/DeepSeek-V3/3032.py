class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        # Precompute the binary lifting tables
        # dp[i][j] will store the (2^j)-th receiver of player i
        # sum_dp[i][j] will store the sum of ids from i to (2^j)-th receiver
        max_j = k.bit_length()
        dp = [[-1] * n for _ in range(max_j)]
        sum_dp = [[0] * n for _ in range(max_j)]
        
        # Initialize the first level (2^0 = 1 step)
        for i in range(n):
            dp[0][i] = receiver[i]
            sum_dp[0][i] = i + receiver[i]
        
        # Fill the dp and sum_dp tables
        for j in range(1, max_j):
            for i in range(n):
                dp[j][i] = dp[j-1][dp[j-1][i]]
                sum_dp[j][i] = sum_dp[j-1][i] + sum_dp[j-1][dp[j-1][i]] - dp[j-1][i]
        
        max_f = 0
        # For each starting player, compute f(x)
        for x in range(n):
            current = x
            total = x
            remaining = k
            j = 0
            while remaining > 0:
                if remaining & 1:
                    total += sum_dp[j][current] - current
                    current = dp[j][current]
                remaining >>= 1
                j += 1
            if total > max_f:
                max_f = total
        return max_f