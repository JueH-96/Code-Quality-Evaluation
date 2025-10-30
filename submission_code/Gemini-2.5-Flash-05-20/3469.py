class Solution:
    def _calculate_max_height_for_start_color(self, red_balls_initial: int, blue_balls_initial: int, start_with_red: bool) -> int:
        """
        Calculates the maximum height of a triangle given initial ball counts
        and a starting color for the first row.

        Args:
            red_balls_initial: The initial count of red balls.
            blue_balls_initial: The initial count of blue balls.
            start_with_red: True if the first row is red, False if blue.

        Returns:
            The maximum height achievable for this starting color pattern.
        """
        r_rem = red_balls_initial
        b_rem = blue_balls_initial
        
        height = 0
        current_row_size = 1
        
        # current_color_is_red indicates if the current row needs red balls
        current_color_is_red = start_with_red
        
        while True:
            if current_color_is_red:
                # Try to form a row with red balls
                if r_rem >= current_row_size:
                    r_rem -= current_row_size
                    height += 1
                else:
                    # Not enough red balls for this row, cannot extend further
                    break 
            else: # current_color_is_blue
                # Try to form a row with blue balls
                if b_rem >= current_row_size:
                    b_rem -= current_row_size
                    height += 1
                else:
                    # Not enough blue balls for this row, cannot extend further
                    break
            
            # Prepare for the next row
            current_row_size += 1
            current_color_is_red = not current_color_is_red # Alternate the color for the next row
            
        return height

    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # Calculate the maximum height if the triangle starts with a red row
        height_starting_red = self._calculate_max_height_for_start_color(red, blue, True)
        
        # Calculate the maximum height if the triangle starts with a blue row
        height_starting_blue = self._calculate_max_height_for_start_color(red, blue, False)
        
        # The overall maximum height is the greater of the two possibilities
        return max(height_starting_red, height_starting_blue)