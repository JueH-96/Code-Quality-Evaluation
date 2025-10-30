from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the costs in descending order to apply the most expensive cuts first
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)

        # Initialize pointers for horizontal and vertical cuts
        i, j = 0, 0
        # Number of segments currently in the vertical and horizontal directions
        h_segments, v_segments = 1, 1
        total_cost = 0

        # Greedily choose the next cut with the higher cost
        while i < len(horizontalCut) and j < len(verticalCut):
            if horizontalCut[i] > verticalCut[j]:
                # Cutting horizontally spans all current vertical segments
                total_cost += horizontalCut[i] * v_segments
                h_segments += 1
                i += 1
            else:
                # Cutting vertically spans all current horizontal segments
                total_cost += verticalCut[j] * h_segments
                v_segments += 1
                j += 1

        # If any horizontal cuts remain, apply them
        while i < len(horizontalCut):
            total_cost += horizontalCut[i] * v_segments
            i += 1

        # If any vertical cuts remain, apply them
        while j < len(verticalCut):
            total_cost += verticalCut[j] * h_segments
            j += 1

        return total_cost