from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        
        # Sort cuts in descending order of cost to greedily pick the most expensive ones first.
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        # Pointers for the two sorted arrays
        h_ptr, v_ptr = 0, 0
        
        # Initially, the cake is one piece, so there is 1 horizontal and 1 vertical segment.
        # A vertical cut will cross `h_segments` pieces.
        # A horizontal cut will cross `v_segments` pieces.
        h_segments, v_segments = 1, 1
        
        total_cost = 0
        
        # Process cuts until both arrays are fully traversed.
        while h_ptr < len(horizontalCut) or v_ptr < len(verticalCut):
            
            # Decide which cut to make next: the one with the higher cost.
            # If one type of cut is exhausted, we must take from the other.
            
            # Check if we should make a horizontal cut.
            # This happens if there are no vertical cuts left, or if the current
            # horizontal cut is more expensive (or equally expensive) than the
            # current vertical cut.
            if v_ptr == len(verticalCut) or \
               (h_ptr < len(horizontalCut) and horizontalCut[h_ptr] >= verticalCut[v_ptr]):
                
                # Make a horizontal cut.
                # The cost is the cut's value times the number of vertical segments it crosses.
                total_cost += horizontalCut[h_ptr] * v_segments
                h_segments += 1
                h_ptr += 1
            else:
                # Make a vertical cut. This branch is taken if there are no horizontal
                # cuts left, or if the vertical cut is more expensive.
                # The cost is the cut's value times the number of horizontal segments it crosses.
                total_cost += verticalCut[v_ptr] * h_segments
                v_segments += 1
                v_ptr += 1
                
        return total_cost