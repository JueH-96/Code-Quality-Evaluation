from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        if n == 0:
            return 0
        
        max_level = 0
        if k > 0:
            max_level = k.bit_length() - 1
        
        # Initialize the binary lifting table
        lift = [[(0, 0) for _ in range(max_level + 1)] for _ in range(n)]
        for x in range(n):
            lift[x][0] = (receiver[x], receiver[x])
        
        # Precompute the binary lifting table for higher levels
        for j in range(1, max_level + 1):
            for x in range(n):
                prev_node, prev_sum = lift[x][j - 1]
                next_node, next_sum = lift[prev_node][j - 1]
                lift[x][j] = (next_node, prev_sum + next_sum)
        
        max_f = 0
        for x in range(n):
            current_node = x
            sum_steps = 0
            steps_remaining = k
            for j in reversed(range(max_level + 1)):
                if steps_remaining >= (1 << j):
                    sum_steps += lift[current_node][j][1]
                    current_node = lift[current_node][j][0]
                    steps_remaining -= (1 << j)
            max_f = max(max_f, x + sum_steps)
        
        return max_f