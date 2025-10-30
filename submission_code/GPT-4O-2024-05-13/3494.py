from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        total_cost = 0
        h_pieces = 1
        v_pieces = 1
        
        i, j = 0, 0
        
        while i < len(horizontalCut) and j < len(verticalCut):
            if horizontalCut[i] >= verticalCut[j]:
                total_cost += horizontalCut[i] * v_pieces
                h_pieces += 1
                i += 1
            else:
                total_cost += verticalCut[j] * h_pieces
                v_pieces += 1
                j += 1
        
        while i < len(horizontalCut):
            total_cost += horizontalCut[i] * v_pieces
            i += 1
        
        while j < len(verticalCut):
            total_cost += verticalCut[j] * h_pieces
            j += 1
        
        return total_cost