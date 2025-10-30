from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cost arrays in descending order so that we always pick the most expensive cut available.
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        total_cost = 0
        # Initially, the cake hasn't been cut, so the number of pieces in each dimension is 1.
        horizontal_pieces = 1  # Number of vertical sections (from vertical cuts)
        vertical_pieces = 1    # Number of horizontal sections (from horizontal cuts)
        
        i, j = 0, 0  # Pointers for horizontalCut and verticalCut arrays
        
        # Process cuts greedily, always picking the cut with the highest cost.
        while i < len(horizontalCut) and j < len(verticalCut):
            if horizontalCut[i] >= verticalCut[j]:
                # A horizontal cut will affect all currently available vertical pieces.
                total_cost += horizontalCut[i] * vertical_pieces
                horizontal_pieces += 1
                i += 1
            else:
                # A vertical cut will affect all currently available horizontal pieces.
                total_cost += verticalCut[j] * horizontal_pieces
                vertical_pieces += 1
                j += 1
                
        # Process any remaining horizontal cuts.
        while i < len(horizontalCut):
            total_cost += horizontalCut[i] * vertical_pieces
            i += 1
        
        # Process any remaining vertical cuts.
        while j < len(verticalCut):
            total_cost += verticalCut[j] * horizontal_pieces
            j += 1
        
        return total_cost