from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        # Compute child1's sum (fixed path)
        child1_sum = 0
        for k in range(n):
            child1_sum += fruits[k][k]
        
        # Compute child2's maximum path
        # Initialize previous DP row for child2
        prev_dp2 = [-float('inf')] * n
        prev_dp2[n-1] = fruits[0][n-1]  # Starting position (0, n-1)
        
        for i in range(1, n):
            curr_dp2 = [-float('inf')] * n
            for j in range(n):
                if j != i:  # Avoid child1's path
                    max_prev = -float('inf')
                    if j - 1 >= 0:
                        max_prev = max(max_prev, prev_dp2[j-1])
                    max_prev = max(max_prev, prev_dp2[j])
                    if j + 1 < n:
                        max_prev = max(max_prev, prev_dp2[j+1])
                    
                    if max_prev != -float('inf'):
                        curr_dp2[j] = fruits[i][j] + max_prev
                else:
                    # On child1's path, do not collect fruits
                    max_prev = -float('inf')
                    if j - 1 >= 0:
                        max_prev = max(max_prev, prev_dp2[j-1])
                    max_prev = max(max_prev, prev_dp2[j])
                    if j + 1 < n:
                        max_prev = max(max_prev, prev_dp2[j+1])
                    
                    if max_prev != -float('inf'):
                        curr_dp2[j] = max_prev
            prev_dp2 = curr_dp2
        
        child2_max = prev_dp2[n-1]
        
        # Compute child3's maximum path
        # Initialize previous DP row for child3
        prev_dp3 = [-float('inf')] * n
        prev_dp3[0] = fruits[n-1][0]  # Starting position (n-1, 0)
        
        for j in range(1, n):
            curr_dp3 = [-float('inf')] * n
            for i in range(n):
                if i != j:  # Avoid child1's path
                    max_prev = -float('inf')
                    if i - 1 >= 0:
                        max_prev = max(max_prev, prev_dp3[i-1])
                    max_prev = max(max_prev, prev_dp3[i])
                    if i + 1 < n:
                        max_prev = max(max_prev, prev_dp3[i+1])
                    
                    if max_prev != -float('inf'):
                        curr_dp3[i] = fruits[i][j] + max_prev
                else:
                    # On child1's path, do not collect fruits
                    max_prev = -float('inf')
                    if i - 1 >= 0:
                        max_prev = max(max_prev, prev_dp3[i-1])
                    max_prev = max(max_prev, prev_dp3[i])
                    if i + 1 < n:
                        max_prev = max(max_prev, prev_dp3[i+1])
                    
                    if max_prev != -float('inf'):
                        curr_dp3[i] = max_prev
            prev_dp3 = curr_dp3
        
        child3_max = prev_dp3[n-1]
        
        # Total sum subtracting the overlapping at (n-1, n-1)
        total_sum = child1_sum + child2_max + child3_max - 2 * fruits[n-1][n-1]
        return total_sum