# Helper function defined inside the method to calculate the height
        # for a given starting color configuration.
        def _calculate_height(r: int, b: int, start_red: bool) -> int:
            """
            Simulates the construction of the triangle row by row and calculates
            the maximum height achievable starting with the specified color.

            Args:
                r: The initial count of red balls.
                b: The initial count of blue balls.
                start_red: A boolean indicating the color of the first row.
                           If True, the first row (row 1) is red.
                           If False, the first row (row 1) is blue.

            Returns:
                The maximum height (number of rows) achievable with this
                starting color configuration.
            """
            height = 0
            row_num = 1  # Represents the current row number being built
                         # and also the number of balls needed for this row.
            current_r = r # Make copies to modify within the simulation
            current_b = b
            # is_red_turn determines if the current row should be red.
            # This depends on the starting color and the row number's parity.
            # We track it directly for simplicity.
            is_red_turn = start_red

            while True:
                balls_needed = row_num

                if is_red_turn:
                    # Current row must be red. Check if enough red balls are available.
                    if current_r >= balls_needed:
                        current_r -= balls_needed # Use the red balls
                        height += 1              # Increment the height (completed one more row)
                        row_num += 1             # Move to the next row
                        is_red_turn = False      # The next row must be blue
                    else:
                        # Not enough red balls to form the current row.
                        # The triangle construction stops here.
                        break
                else: # Blue turn
                    # Current row must be blue. Check if enough blue balls are available.
                    if current_b >= balls_needed:
                        current_b -= balls_needed # Use the blue balls
                        height += 1              # Increment the height
                        row_num += 1             # Move to the next row
                        is_red_turn = True       # The next row must be red
                    else:
                        # Not enough blue balls to form the current row.
                        # The triangle construction stops here.
                        break
            # Return the total height achieved for this starting configuration.
            return height

        # Case 1: Simulate the triangle construction starting with red for the first row.
        height_start_red = _calculate_height(red, blue, True)

        # Case 2: Simulate the triangle construction starting with blue for the first row.
        height_start_blue = _calculate_height(red, blue, False)

        # The overall maximum height is the maximum height achieved from either
        # starting configuration (starting with red or starting with blue).
        return max(height_start_red, height_start_blue)