from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        if n == 0:
            return 0
        
        # Calculate sum of diagonal cells
        sum_diagonal = sum(fruits[i][i] for i in range(n))
        
        # Dynamic programming for the second child's path
        prev_second = [-float('inf')] * n
        prev_second[-1] = fruits[0][-1]  # Starting at (0, n-1)
        for k in range(1, n):
            current_second = [-float('inf')] * n
            for j in range(n):
                if j + 1 < n and prev_second[j + 1] != -float('inf'):
                    current_second[j] = max(current_second[j], prev_second[j + 1] + fruits[k][j + 1])
                if prev_second[j] != -float('inf'):
                    current_second[j] = max(current_second[j], prev_second[j] + fruits[k][j])
                if j - 1 >= 0 and prev_second[j - 1] != -float('inf'):
                    current_second[j] = max(current_second[j], prev_second[j - 1] + fruits[k][j - 1])
            prev_second = current_second
        sum_second = prev_second[-1]
        
        # Dynamic programming for the third child's path
        prev_third = [-float('inf')] * n
        prev_third[-1] = fruits[-1][0]  # Starting at (n-1, 0)
        for k in range(1, n):
            current_third = [-float('inf')] * n
            for i in range(n):
                if i + 1 < n and prev_third[i + 1] != -float('inf'):
                    current_third[i] = max(current_third[i], prev_third[i + 1] + fruits[i + 1][k])
                if prev_third[i] != -float('inf'):
                    current_third[i] = max(current_third[i], prev_third[i] + fruits[i][k])
                if i - 1 >= 0 and prev_third[i - 1] != -float('inf'):
                    current_third[i] = max(current_third[i], prev_third[i - 1] + fruits[i - 1][k])
            prev_third = current_third
        sum_third = prev_third[-1]
        
        # Calculate the total sum, subtracting the overlapping end cell twice
        total = sum_diagonal + sum_second + sum_third - 2 * fruits[-1][-1]
        return total