class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        
        # Helper function to simulate building the triangle for a given starting color
        # and initial ball counts.
        # Parameters:
        #   r_balls_initial: count of red balls available at the start of this scenario.
        #   b_balls_initial: count of blue balls available at the start of this scenario.
        #   first_row_is_red: boolean, True if the first row is red, False if blue.
        # Returns:
        #   The maximum height achievable with this starting configuration.
        def _calculate_height_for_scenario(r_balls_initial: int, b_balls_initial: int, first_row_is_red: bool) -> int:
            height = 0
            # current_row_level also represents the number of balls needed for this row.
            current_row_level = 1 
            
            # Make copies of ball counts to modify during simulation for this scenario.
            remaining_red = r_balls_initial
            remaining_blue = b_balls_initial
            
            # Determines if the current row to be built should be red or blue.
            current_row_should_be_red = first_row_is_red
            
            while True:
                balls_needed_for_current_row = current_row_level
                
                if current_row_should_be_red:
                    if remaining_red >= balls_needed_for_current_row:
                        remaining_red -= balls_needed_for_current_row
                        # Next row must be blue
                        current_row_should_be_red = False 
                    else:
                        # Not enough red balls for the current row, so this scenario ends.
                        break 
                else: # Current row should be blue
                    if remaining_blue >= balls_needed_for_current_row:
                        remaining_blue -= balls_needed_for_current_row
                        # Next row must be red
                        current_row_should_be_red = True 
                    else:
                        # Not enough blue balls for the current row, so this scenario ends.
                        break
                
                height += 1
                # Move to the next level, which will require one more ball.
                current_row_level += 1 
            
            return height

        # Scenario 1: The first row (1 ball) is red.
        # In this scenario, odd-numbered rows (1, 3, 5...) are red,
        # and even-numbered rows (2, 4, 6...) are blue.
        height_if_start_red = _calculate_height_for_scenario(red, blue, True)
        
        # Scenario 2: The first row (1 ball) is blue.
        # In this scenario, odd-numbered rows (1, 3, 5...) are blue,
        # and even-numbered rows (2, 4, 6...) are red.
        height_if_start_blue = _calculate_height_for_scenario(red, blue, False)
        
        # The overall maximum height is the greater of the heights achievable 
        # from these two distinct scenarios.
        return max(height_if_start_red, height_if_start_blue)