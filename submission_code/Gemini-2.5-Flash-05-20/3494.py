from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort costs in descending order. This is crucial for the greedy strategy,
        # ensuring that we always process the most expensive available cut first.
        # The logic is that higher costs should be multiplied by smaller multipliers
        # (number of existing pieces) which are available earlier in the process.
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)

        total_cost = 0
        
        # 'horizontal_pieces' represents the number of cake segments currently separated
        # by horizontal cuts. Initially, the entire cake is one piece (1 segment).
        # Each horizontal cut adds one more horizontal segment.
        horizontal_pieces = 1 
        
        # 'vertical_pieces' represents the number of cake segments currently separated
        # by vertical cuts. Initially, the entire cake is one piece (1 segment).
        # Each vertical cut adds one more vertical segment.
        vertical_pieces = 1

        # Pointers to keep track of the next highest cost cut to consider from each list.
        h_ptr = 0
        v_ptr = 0

        # Continue as long as there are horizontal cuts (m-1 needed) or
        # vertical cuts (n-1 needed) remaining to be made.
        while h_ptr < m - 1 or v_ptr < n - 1:
            # Decision logic based on the greedy strategy:
            # 1. If all horizontal cuts are made, only vertical cuts remain.
            # 2. If all vertical cuts are made, only horizontal cuts remain.
            # 3. If both are available, pick the one with the highest cost.
            
            if h_ptr == m - 1: # All horizontal cuts exhausted
                # We must make a vertical cut.
                current_cost = verticalCut[v_ptr]
                # This vertical cut will affect 'horizontal_pieces' distinct segments.
                total_cost += current_cost * horizontal_pieces
                vertical_pieces += 1 # Increment count of vertical segments
                v_ptr += 1           # Move to the next vertical cut
            elif v_ptr == n - 1: # All vertical cuts exhausted
                # We must make a horizontal cut.
                current_cost = horizontalCut[h_ptr]
                # This horizontal cut will affect 'vertical_pieces' distinct segments.
                total_cost += current_cost * vertical_pieces
                horizontal_pieces += 1 # Increment count of horizontal segments
                h_ptr += 1             # Move to the next horizontal cut
            elif horizontalCut[h_ptr] >= verticalCut[v_ptr]:
                # The current most expensive horizontal cut is greater than or equal to
                # the current most expensive vertical cut. Choose the horizontal cut.
                current_cost = horizontalCut[h_ptr]
                total_cost += current_cost * vertical_pieces
                horizontal_pieces += 1
                h_ptr += 1
            else: # verticalCut[v_ptr] > horizontalCut[h_ptr]
                # The current most expensive vertical cut is strictly greater. Choose it.
                current_cost = verticalCut[v_ptr]
                total_cost += current_cost * horizontal_pieces
                vertical_pieces += 1
                v_ptr += 1
                
        return total_cost