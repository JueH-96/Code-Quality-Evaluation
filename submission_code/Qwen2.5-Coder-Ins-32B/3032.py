from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_bits = k.bit_length()
        
        # dp[i][j] will store the 2^j-th receiver of player i
        # sum_dp[i][j] will store the sum of ids of players received the ball in 2^j passes starting from player i
        dp = [[-1] * max_bits for _ in range(n)]
        sum_dp = [[0] * max_bits for _ in range(n)]
        
        for i in range(n):
            dp[i][0] = receiver[i]
            sum_dp[i][0] = receiver[i]
        
        for j in range(1, max_bits):
            for i in range(n):
                dp[i][j] = dp[dp[i][j-1]][j-1]
                sum_dp[i][j] = sum_dp[i][j-1] + sum_dp[dp[i][j-1]][j-1]
        
        def get_kth_receiver_and_sum(start, k):
            current = start
            total_sum = start
            for j in range(max_bits):
                if k & (1 << j):
                    total_sum += sum_dp[current][j]
                    current = dp[current][j]
            return total_sum
        
        max_value = 0
        for i in range(n):
            max_value = max(max_value, get_kth_receiver_and_sum(i, k))
        
        return max_value