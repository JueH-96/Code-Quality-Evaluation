from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_level = 35  # Sufficient to cover up to 2^34 steps, which is ~1.7e10
        
        # Initialize up and sum_val tables
        up = [[0] * n for _ in range(max_level)]
        sum_val = [[0] * n for _ in range(max_level)]
        
        # Fill level 0
        for x in range(n):
            up[0][x] = receiver[x]
            sum_val[0][x] = receiver[x]
        
        # Fill higher levels
        for i in range(1, max_level):
            for x in range(n):
                prev = up[i-1][x]
                up[i][x] = up[i-1][prev]
                sum_val[i][x] += sum_val[i-1][x] + sum_val[i-1][prev]
        
        max_total = 0
        for x in range(n):
            current = x
            total_steps_sum = 0
            remaining = k
            # Decompose remaining steps into powers of two
            for i in range(max_level-1, -1, -1):
                if (1 << i) <= remaining:
                    total_steps_sum += sum_val[i][current]
                    current = up[i][current]
                    remaining -= (1 << i)
            current_f = x + total_steps_sum
            if current_f > max_total:
                max_total = current_f
        
        return max_total