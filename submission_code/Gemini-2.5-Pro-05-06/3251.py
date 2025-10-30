from typing import List

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        
        # Initialize variables to keep track of the maximum squared diagonal
        # and the area of the rectangle corresponding to that diagonal.
        # Using 0 for initialization is safe because lengths and widths are >= 1,
        # so any valid rectangle will have a squared diagonal >= 2 and area >= 1.
        max_diag_sq = 0
        result_area = 0 

        # Iterate through each rectangle's dimensions
        for dim in dimensions:
            length = dim[0]
            width = dim[1]

            # Calculate the square of the diagonal length for the current rectangle.
            # d^2 = length^2 + width^2
            current_diag_sq = length * length + width * width
            
            # Calculate the area of the current rectangle.
            # Area = length * width
            current_area = length * width

            # Check if this rectangle has a longer diagonal than previously found
            if current_diag_sq > max_diag_sq:
                # If so, this rectangle has a new longest diagonal.
                # Update both max_diag_sq and result_area.
                max_diag_sq = current_diag_sq
                result_area = current_area
            elif current_diag_sq == max_diag_sq:
                # If the diagonal is the same length as the current maximum,
                # we need to choose the one with the maximum area among them.
                # Update result_area if current_area is larger.
                result_area = max(result_area, current_area)
            # If current_diag_sq < max_diag_sq, this rectangle's diagonal is shorter,
            # so we ignore it based on the problem's criteria.
        
        return result_area