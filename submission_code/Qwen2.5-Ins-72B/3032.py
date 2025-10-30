from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_value = 0
        
        # Precompute the next player and the sum of ids for 2^j passes
        next_player = [[0] * 64 for _ in range(n)]
        sum_ids = [[0] * 64 for _ in range(n)]
        
        for i in range(n):
            next_player[i][0] = receiver[i]
            sum_ids[i][0] = receiver[i]
        
        for j in range(1, 64):
            for i in range(n):
                next_player[i][j] = next_player[next_player[i][j-1]][j-1]
                sum_ids[i][j] = sum_ids[i][j-1] + sum_ids[next_player[i][j-1]][j-1]
        
        # Function to compute f(x) for a given starting player x
        def compute_f(x: int) -> int:
            result = x
            current = x
            for j in range(64):
                if k & (1 << j):
                    result += sum_ids[current][j]
                    current = next_player[current][j]
            return result
        
        # Find the maximum value of f(x) for all starting players
        for x in range(n):
            max_value = max(max_value, compute_f(x))
        
        return max_value