from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        i = j = 0
        total = 0
        num_h = 0  # Number of horizontal cuts made
        num_v = 0  # Number of vertical cuts made
        
        while i < len(horizontalCut) and j < len(verticalCut):
            if horizontalCut[i] > verticalCut[j]:
                # Process horizontal cut
                total += horizontalCut[i] * (num_v + 1)
                num_h += 1
                i += 1
            else:
                # Process vertical cut
                total += verticalCut[j] * (num_h + 1)
                num_v += 1
                j += 1
        
        # Process remaining horizontal cuts
        while i < len(horizontalCut):
            total += horizontalCut[i] * (num_v + 1)
            i += 1
        
        # Process remaining vertical cuts
        while j < len(verticalCut):
            total += verticalCut[j] * (num_h + 1)
            j += 1
        
        return total