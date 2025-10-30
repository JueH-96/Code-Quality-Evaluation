from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_level = 34  # Since 2^34 is approximately 1.7e10, which covers k up to 1e10
        jumps = [[(0, 0) for _ in range(n)] for _ in range(max_level + 1)]
        
        # Initialize the base case (j=0: 1 step)
        for x in range(n):
            jumps[0][x] = (receiver[x], receiver[x])
        
        # Build the binary lifting table for higher levels
        for j in range(1, max_level + 1):
            for x in range(n):
                mid_node, mid_sum = jumps[j-1][x]
                next_node, next_sum = jumps[j-1][mid_node]
                jumps[j][x] = (next_node, mid_sum + next_sum)
        
        max_sum = 0
        for x in range(n):
            current_node = x
            remaining = k
            total = 0
            # Decompose k into binary components and accumulate the sum
            for j in range(max_level, -1, -1):
                if (1 << j) <= remaining:
                    total += jumps[j][current_node][1]
                    current_node = jumps[j][current_node][0]
                    remaining -= (1 << j)
            max_sum = max(max_sum, x + total)
        
        return max_sum