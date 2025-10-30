from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cut costs in descending order for greedy selection
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        # Number of current segments in each direction
        h_segments = 1  # how many horizontal strips we have
        v_segments = 1  # how many vertical strips we have
        
        i = 0  # pointer into horizontalCut
        j = 0  # pointer into verticalCut
        total_cost = 0
        
        # Greedily pick the highest cost cut available
        while i < len(horizontalCut) and j < len(verticalCut):
            if horizontalCut[i] > verticalCut[j]:
                # Make a horizontal cut across all current vertical segments
                total_cost += horizontalCut[i] * v_segments
                h_segments += 1
                i += 1
            else:
                # Make a vertical cut across all current horizontal segments
                total_cost += verticalCut[j] * h_segments
                v_segments += 1
                j += 1
        
        # If any horizontal cuts remain, apply them
        while i < len(horizontalCut):
            total_cost += horizontalCut[i] * v_segments
            h_segments += 1
            i += 1
        
        # If any vertical cuts remain, apply them
        while j < len(verticalCut):
            total_cost += verticalCut[j] * h_segments
            v_segments += 1
            j += 1
        
        return total_cost