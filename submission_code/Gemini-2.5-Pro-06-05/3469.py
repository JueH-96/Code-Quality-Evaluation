class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        
        def _calculate_height(first_balls: int, second_balls: int) -> int:
            """
            Calculates the height of a triangle by simulating its construction.
            - Odd rows use balls from the `first_balls` count.
            - Even rows use balls from the `second_balls` count.
            """
            height = 0
            row_num = 1
            while True:
                if row_num % 2 == 1:
                    # Odd row needs `row_num` balls of the first color.
                    if first_balls >= row_num:
                        first_balls -= row_num
                    else:
                        break  # Not enough balls to build this row.
                else:
                    # Even row needs `row_num` balls of the second color.
                    if second_balls >= row_num:
                        second_balls -= row_num
                    else:
                        break  # Not enough balls to build this row.
                
                height += 1
                row_num += 1
            return height

        # Scenario 1: Start with red. Odd rows are red, even rows are blue.
        height_with_red_start = _calculate_height(red, blue)
        
        # Scenario 2: Start with blue. Odd rows are blue, even rows are red.
        height_with_blue_start = _calculate_height(blue, red)
        
        # The answer is the maximum height from the two possible scenarios.
        return max(height_with_red_start, height_with_blue_start)