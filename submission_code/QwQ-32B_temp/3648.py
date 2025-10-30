from typing import List
import sys

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        if n == 0:
            return 0
        
        # Compute first path's total
        first_total = 0
        for i in range(n):
            first_total += fruits[i][i]
        
        # Compute second child's max path
        second_dp = [-sys.maxsize] * n
        second_dp[n-1] = fruits[0][n-1]
        for k in range(1, n):
            new_dp = [-sys.maxsize] * n
            for j in range(n):
                max_prev = -sys.maxsize
                for dj in (-1, 0, 1):
                    prev_j = j + dj
                    if 0 <= prev_j < n:
                        if second_dp[prev_j] > max_prev:
                            max_prev = second_dp[prev_j]
                if max_prev != -sys.maxsize:
                    current_val = max_prev + (fruits[k][j] if j != k else 0)
                    new_dp[j] = current_val
            second_dp = new_dp
        
        second_max = second_dp[n-1]
        
        # Compute third child's max path
        third_dp = [-sys.maxsize] * n
        third_dp[n-1] = fruits[n-1][0]
        for k in range(1, n):
            new_dp = [-sys.maxsize] * n
            for i in range(n):
                max_prev = -sys.maxsize
                for di in (-1, 0, 1):
                    prev_i = i + di
                    if 0 <= prev_i < n:
                        if third_dp[prev_i] > max_prev:
                            max_prev = third_dp[prev_i]
                if max_prev != -sys.maxsize:
                    current_val = max_prev + (fruits[i][k] if i != k else 0)
                    new_dp[i] = current_val
            third_dp = new_dp
        
        third_max = third_dp[n-1]
        
        return first_total + second_max + third_max