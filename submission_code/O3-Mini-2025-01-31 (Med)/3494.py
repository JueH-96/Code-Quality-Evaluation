from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cost lists in descending order.
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        cost = 0
        horizontalPieces = 1  # Initially, there is 1 vertical segment.
        verticalPieces = 1    # Initially, there is 1 horizontal segment.
        i = 0  # index for horizontal cuts
        j = 0  # index for vertical cuts
        
        # Process both lists greedily.
        while i < len(horizontalCut) and j < len(verticalCut):
            if horizontalCut[i] >= verticalCut[j]:
                # When performing a horizontal cut, it spans across all vertical segments.
                cost += horizontalCut[i] * verticalPieces
                horizontalPieces += 1  # Now one more horizontal cut is made.
                i += 1
            else:
                # When performing a vertical cut, it spans across all horizontal segments.
                cost += verticalCut[j] * horizontalPieces
                verticalPieces += 1  # Now one more vertical cut is made.
                j += 1
        
        # If any horizontal cuts remain, process them.
        while i < len(horizontalCut):
            cost += horizontalCut[i] * verticalPieces
            horizontalPieces += 1
            i += 1
        
        # If any vertical cuts remain, process them.
        while j < len(verticalCut):
            cost += verticalCut[j] * horizontalPieces
            verticalPieces += 1
            j += 1
        
        return cost