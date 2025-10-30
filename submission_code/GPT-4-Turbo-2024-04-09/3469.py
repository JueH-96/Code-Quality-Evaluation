class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # Initialize variables
        height = 0
        current_row = 1
        
        # Loop until we run out of balls
        while True:
            # Check if we can place the current row with the current color
            if height % 2 == 0:  # Even height, use more abundant color
                if red >= blue:
                    if red >= current_row:
                        red -= current_row
                    else:
                        break
                else:
                    if blue >= current_row:
                        blue -= current_row
                    else:
                        break
            else:  # Odd height, use the less abundant color
                if red < blue:
                    if red >= current_row:
                        red -= current_row
                    else:
                        break
                else:
                    if blue >= current_row:
                        blue -= current_row
                    else:
                        break
            
            # Successfully placed the row, increment height and prepare for the next row
            height += 1
            current_row += 1
        
        return height