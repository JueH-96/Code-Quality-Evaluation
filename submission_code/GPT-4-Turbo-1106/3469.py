class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # Initialize the maximum height to 0
        max_height = 0
        # Initialize the current row to 1
        current_row = 1
        
        # Loop until there are not enough balls to form the next row
        while red >= current_row or blue >= current_row:
            # If the current row is odd, we will use red balls
            if current_row % 2 == 1:
                # If there are enough red balls for the current row
                if red >= current_row:
                    # Use the red balls and increment the height
                    red -= current_row
                    max_height += 1
                # If there are not enough red balls, break the loop
                else:
                    break
            # If the current row is even, we will use blue balls
            else:
                # If there are enough blue balls for the current row
                if blue >= current_row:
                    # Use the blue balls and increment the height
                    blue -= current_row
                    max_height += 1
                # If there are not enough blue balls, break the loop
                else:
                    break
            # Move to the next row
            current_row += 1
        
        # Return the maximum height of the triangle
        return max_height