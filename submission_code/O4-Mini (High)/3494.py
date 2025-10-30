from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cut costs in descending order
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        # Initial number of pieces in each direction
        h_pieces = 1
        v_pieces = 1
        
        i, j = 0, 0
        total_cost = 0
        
        # Greedily pick the largest remaining cut cost
        while i < len(horizontalCut) and j < len(verticalCut):
            if horizontalCut[i] >= verticalCut[j]:
                # Make a horizontal cut across all current vertical pieces
                total_cost += horizontalCut[i] * v_pieces
                h_pieces += 1
                i += 1
            else:
                # Make a vertical cut across all current horizontal pieces
                total_cost += verticalCut[j] * h_pieces
                v_pieces += 1
                j += 1
        
        # Process any remaining horizontal cuts
        while i < len(horizontalCut):
            total_cost += horizontalCut[i] * v_pieces
            i += 1
            h_pieces += 1
        
        # Process any remaining vertical cuts
        while j < len(verticalCut):
            total_cost += verticalCut[j] * h_pieces
            j += 1
            v_pieces += 1
        
        return total_cost