import math
from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        
        # Determine the initial search range for Y.
        # The minimum Y coordinate will be the lowest bottom_edge of any square.
        # The maximum Y coordinate will be the highest top_edge of any square.
        min_possible_y = float('inf')
        max_possible_y_plus_l = float('-inf')

        for x, y, l in squares:
            # Convert integer coordinates and lengths to float to avoid precision issues
            # during calculations, especially with large values.
            y_f = float(y)
            l_f = float(l)
            min_possible_y = min(min_possible_y, y_f)
            max_possible_y_plus_l = max(max_possible_y_plus_l, y_f + l_f)
        
        # Initialize binary search bounds.
        low = min_possible_y
        high = max_possible_y_plus_l
        
        # Perform binary search for a fixed number of iterations.
        # 100 iterations are typically sufficient for double-precision floating points
        # to achieve high accuracy (e.g., 10^-5 or better) over typical ranges.
        for _ in range(100): 
            mid = (low + high) / 2.0
            
            area_below_mid = 0.0
            area_above_mid = 0.0

            # Iterate through all squares to calculate total areas below and above 'mid' line.
            for x_s, y_s, l_s in squares:
                y_s_f = float(y_s)
                l_s_f = float(l_s)
                
                bottom_edge = y_s_f
                top_edge = y_s_f + l_s_f
                
                # Calculate the area contributed by the current square that is below the line 'mid'.
                if mid <= bottom_edge:
                    # The line 'mid' is below or exactly at the square's bottom edge.
                    # No part of this square is below the line.
                    current_square_area_below = 0.0
                elif mid >= top_edge:
                    # The line 'mid' is above or exactly at the square's top edge.
                    # The entire square is below the line.
                    current_square_area_below = l_s_f * l_s_f
                else: # bottom_edge < mid < top_edge
                    # The line 'mid' cuts through the square.
                    # The height of the portion below 'mid' is 'mid - bottom_edge'.
                    height_below = mid - bottom_edge
                    current_square_area_below = l_s_f * height_below
                
                area_below_mid += current_square_area_below

                # Calculate the area contributed by the current square that is above the line 'mid'.
                if mid >= top_edge:
                    # The line 'mid' is above or exactly at the square's top edge.
                    # No part of this square is above the line.
                    current_square_area_above = 0.0
                elif mid <= bottom_edge:
                    # The line 'mid' is below or exactly at the square's bottom edge.
                    # The entire square is above the line.
                    current_square_area_above = l_s_f * l_s_f
                else: # bottom_edge < mid < top_edge
                    # The line 'mid' cuts through the square.
                    # The height of the portion above 'mid' is 'top_edge - mid'.
                    height_above = top_edge - mid
                    current_square_area_above = l_s_f * height_above
                
                area_above_mid += current_square_area_above
            
            # Adjust the binary search range based on the comparison of areas.
            # We are looking for Y such that area_below(Y) = area_above(Y).
            # The function f(Y) = area_below(Y) - area_above(Y) is monotonically non-decreasing.
            if area_below_mid < area_above_mid:
                # If area below is less than area above, it means 'mid' is too low.
                # We need to increase Y to balance the areas.
                low = mid
            else: # area_below_mid >= area_above_mid
                # If area below is greater than or equal to area above, 'mid' is at or too high.
                # We potentially need to decrease Y or 'mid' is a valid candidate.
                # By setting high = mid, we search the lower half for the minimum Y.
                high = mid
                
        # After 100 iterations, 'low' and 'high' will have converged to the desired y-coordinate
        # within the required precision. Either can be returned.
        return high