from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_log = 40
        
        next_table = [[0] * n for _ in range(max_log)]
        sum_table = [[0] * n for _ in range(max_log)]
        
        for i in range(n):
            next_table[0][i] = receiver[i]
            sum_table[0][i] = receiver[i]
        
        for j in range(1, max_log):
            for i in range(n):
                mid = next_table[j-1][i]
                next_table[j][i] = next_table[j-1][mid]
                sum_table[j][i] = sum_table[j-1][i] + sum_table[j-1][mid]
        
        best = -10**18
        
        for x in range(n):
            total = x
            cur = x
            rem = k
            for j in range(max_log-1, -1, -1):
                if rem >= (1 << j):
                    total += sum_table[j][cur]
                    cur = next_table[j][cur]
                    rem -= (1 << j)
            if total > best:
                best = total
        
        return best