from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        sum_diagonal = sum(fruits[i][i] for i in range(n))
        
        # Compute max_second_sum
        dp_second = [[-1 for _ in range(n)] for __ in range(n)]
        dp_second[0][n-1] = fruits[0][n-1] if 0 != n-1 else 0
        for k in range(1, n):
            for j in range(n):
                max_prev = -1
                for dj in [-1, 0, 1]:
                    prev_j = j + dj
                    if 0 <= prev_j < n and dp_second[k-1][prev_j] != -1:
                        max_prev = max(max_prev, dp_second[k-1][prev_j])
                if max_prev != -1:
                    current = fruits[k][j] if j != k else 0
                    dp_second[k][j] = max_prev + current
        
        max_second = dp_second[n-1][n-1] if dp_second[n-1][n-1] != -1 else 0
        
        # Compute max_third_sum
        dp_third = [[-1 for _ in range(n)] for __ in range(n)]
        dp_third[0][n-1] = fruits[n-1][0] if n-1 != 0 else 0
        for k in range(1, n):
            for i in range(n):
                max_prev = -1
                for di in [-1, 0, 1]:
                    prev_i = i + di
                    if 0 <= prev_i < n and dp_third[k-1][prev_i] != -1:
                        max_prev = max(max_prev, dp_third[k-1][prev_i])
                if max_prev != -1:
                    current = fruits[i][k] if i != k else 0
                    dp_third[k][i] = max_prev + current
        
        max_third = dp_third[n-1][n-1] if dp_third[n-1][n-1] != -1 else 0
        
        return sum_diagonal + max_second + max_third