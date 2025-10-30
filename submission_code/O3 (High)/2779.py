from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        # Array to keep current colors, 0 = uncoloured
        nums = [0] * n
        # Current number of adjacent equal‐coloured pairs
        adj_equal = 0  
        res = []
        
        for idx, color in queries:
            prev = nums[idx]
            
            # If index is already coloured, remove its previous contribution
            if idx - 1 >= 0 and nums[idx - 1] == prev and prev != 0:
                adj_equal -= 1
            if idx + 1 < n and nums[idx + 1] == prev and prev != 0:
                adj_equal -= 1
            
            # Apply new colour
            nums[idx] = color
            
            # Add new contribution
            if idx - 1 >= 0 and nums[idx - 1] == color and color != 0:
                adj_equal += 1
            if idx + 1 < n and nums[idx + 1] == color and color != 0:
                adj_equal += 1
            
            res.append(adj_equal)
        
        return res