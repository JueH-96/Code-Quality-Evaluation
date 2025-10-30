class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        sum_main = sum(fruits[i][i] for i in range(n))
        
        # Compute DP for child2 starting at (0, n-1)
        prev2 = [float('-inf')] * n
        prev2[n-1] = fruits[0][n-1]  # Initial position (0, n-1)
        for m in range(1, n):
            curr2 = [float('-inf')] * n
            for j in range(n):
                max_prev = float('-inf')
                # Check possible previous positions j-1, j, j+1
                for pj in [j-1, j, j+1]:
                    if 0 <= pj < n:
                        if prev2[pj] > max_prev:
                            max_prev = prev2[pj]
                if max_prev != float('-inf'):
                    if m != j:
                        curr2[j] = max_prev + fruits[m][j]
                    else:
                        curr2[j] = max_prev
                else:
                    curr2[j] = float('-inf')
            prev2 = curr2
        
        # Compute DP for child3 starting at (n-1, 0)
        prev3 = [float('-inf')] * n
        prev3[n-1] = fruits[n-1][0]  # Initial position (n-1, 0)
        for m in range(1, n):
            curr3 = [float('-inf')] * n
            for i in range(n):
                max_prev = float('-inf')
                # Check possible previous positions i-1, i, i+1
                for pi in [i-1, i, i+1]:
                    if 0 <= pi < n:
                        if prev3[pi] > max_prev:
                            max_prev = prev3[pi]
                if max_prev != float('-inf'):
                    if i != m:
                        curr3[i] = max_prev + fruits[i][m]
                    else:
                        curr3[i] = max_prev
                else:
                    curr3[i] = float('-inf')
            prev3 = curr3
        
        return sum_main + prev2[-1] + prev3[-1]