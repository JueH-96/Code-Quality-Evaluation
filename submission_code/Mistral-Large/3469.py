class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # Initialize variables to keep track of the remaining balls and the current row number
        remaining_red = red
        remaining_blue = blue
        current_row = 1

        # Determine the color of the first row
        if red > blue:
            current_color = 'red'
        else:
            current_color = 'blue'

        # Loop until we cannot form a new row with the remaining balls
        while True:
            if current_color == 'red':
                if remaining_red >= current_row:
                    remaining_red -= current_row
                    current_color = 'blue'
                else:
                    break
            else:
                if remaining_blue >= current_row:
                    remaining_blue -= current_row
                    current_color = 'red'
                else:
                    break
            current_row += 1

        # The maximum height is the last completed row number
        return current_row - 1