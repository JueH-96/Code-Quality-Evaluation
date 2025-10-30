class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        diagonal_sum = sum(fruits[i][i] for i in range(n))
        
        # Dynamic Programming for child 2 (starting from (0, n-1))
        prev_c2 = [-float('inf')] * n
        prev_c2[n-1] = fruits[0][n-1]  # Starting position (0, n-1)
        
        for k in range(1, n):
            curr = [-float('inf')] * n
            for j in range(n):
                max_prev = -float('inf')
                # Check previous positions j-1, j, j+1
                for dj in (-1, 0, 1):
                    pj = j - dj
                    if 0 <= pj < n:
                        if prev_c2[pj] > max_prev:
                            max_prev = prev_c2[pj]
                if max_prev != -float('inf'):
                    if k != j:
                        curr[j] = max_prev + fruits[k][j]
                    else:
                        curr[j] = max_prev
            prev_c2 = curr
        
        c2_sum = prev_c2[-1] if prev_c2[-1] != -float('inf') else 0
        
        # Dynamic Programming for child 3 (starting from (n-1, 0))
        prev_c3 = [-float('inf')] * n
        prev_c3[n-1] = fruits[n-1][0]  # Starting position (n-1, 0)
        
        for k in range(1, n):
            curr = [-float('inf')] * n
            for i in range(n):
                max_prev = -float('inf')
                # Check previous positions i-1, i, i+1
                for di in (-1, 0, 1):
                    pi = i - di
                    if 0 <= pi < n:
                        if prev_c3[pi] > max_prev:
                            max_prev = prev_c3[pi]
                if max_prev != -float('inf'):
                    if i != k:
                        curr[i] = max_prev + fruits[i][k]
                    else:
                        curr[i] = max_prev
            prev_c3 = curr
        
        c3_sum = prev_c3[-1] if prev_c3[-1] != -float('inf') else 0
        
        return diagonal_sum + c2_sum + c3_sum