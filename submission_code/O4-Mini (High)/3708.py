from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        result = []
        take = True  # flag to decide whether to take the current cell
        
        for i, row in enumerate(grid):
            # determine traversal direction: left-to-right on even rows, right-to-left on odd rows
            if i % 2 == 0:
                cols = range(len(row))
            else:
                cols = range(len(row) - 1, -1, -1)
            
            for j in cols:
                if take:
                    result.append(row[j])
                take = not take  # flip the flag for the next cell
                
        return result