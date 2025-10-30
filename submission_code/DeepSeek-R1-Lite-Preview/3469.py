class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # Calculate the maximum possible height
        total_balls = red + blue
        h_max = int((2 * total_balls)**0.5)
        while (h_max * (h_max + 1)) // 2 > total_balls:
            h_max -= 1
        
        max_height = 0
        for h in range(1, h_max + 1):
            # Starting with red
            red_rows = (h + 1) // 2
            blue_rows = h // 2
            sum_red_start = red_rows * red_rows  # Sum of first red_rows odd numbers
            sum_blue_start = blue_rows * (blue_rows + 1)  # Sum of first blue_rows even numbers
            if red >= sum_red_start and blue >= sum_blue_start:
                max_height = h
                continue  # No need to check starting with blue for this h
            
            # Starting with blue
            blue_rows_start = (h + 1) // 2
            red_rows_start = h // 2
            sum_blue_start_alt = blue_rows_start * blue_rows_start  # Sum of first blue_rows_start odd numbers
            sum_red_start_alt = red_rows_start * (red_rows_start + 1)  # Sum of first red_rows_start even numbers
            if blue >= sum_blue_start_alt and red >= sum_red_start_alt:
                max_height = h
                continue  # No need to check further for this h
        
        return max_height