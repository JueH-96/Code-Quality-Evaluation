from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts
        horizontalCut.sort()
        verticalCut.sort()
        
        # Calculate the maximum gap between horizontal cuts
        max_h_gap = max(horizontalCut[0], m - horizontalCut[-1])
        for i in range(1, len(horizontalCut)):
            max_h_gap = max(max_h_gap, horizontalCut[i] - horizontalCut[i - 1])
        
        # Calculate the maximum gap between vertical cuts
        max_v_gap = max(verticalCut[0], n - verticalCut[-1])
        for i in range(1, len(verticalCut)):
            max_v_gap = max(max_v_gap, verticalCut[i] - verticalCut[i - 1])
        
        # The minimum cost is the product of the maximum gaps
        return max_h_gap * len(verticalCut) + max_v_gap * len(horizontalCut)