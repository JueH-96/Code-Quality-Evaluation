class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        
        # Precompute the 2^j-th receiver for each player
        max_j = 60  # Since k <= 10^10, 2^60 > 10^10
        dp = [[0] * n for _ in range(max_j)]
        sum_dp = [[0] * n for _ in range(max_j)]
        
        # Initialize for j = 0
        for i in range(n):
            dp[0][i] = receiver[i]
            sum_dp[0][i] = receiver[i]
        
        # Fill the dp and sum_dp tables
        for j in range(1, max_j):
            for i in range(n):
                dp[j][i] = dp[j-1][dp[j-1][i]]
                sum_dp[j][i] = sum_dp[j-1][i] + sum_dp[j-1][dp[j-1][i]]
        
        # Function to calculate f(x) for a given starting player x
        def calculate_f(x):
            total_sum = x
            current = x
            remaining_k = k
            j = 0
            while remaining_k > 0:
                if remaining_k & 1:
                    total_sum += sum_dp[j][current]
                    current = dp[j][current]
                remaining_k >>= 1
                j += 1
            return total_sum
        
        # Find the maximum f(x) over all starting players
        max_value = 0
        for x in range(n):
            max_value = max(max_value, calculate_f(x))
        
        return max_value