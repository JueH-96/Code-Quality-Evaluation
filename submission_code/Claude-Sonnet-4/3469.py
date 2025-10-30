class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def max_height_with_first_color(first_color_count, second_color_count):
            height = 0
            first_used = 0  # balls used for odd rows (1,3,5,...)
            second_used = 0  # balls used for even rows (2,4,6,...)
            
            for row in range(1, 101):  # max possible height is limited by constraint
                if row % 2 == 1:  # odd row - uses first color
                    if first_used + row <= first_color_count:
                        first_used += row
                        height = row
                    else:
                        break
                else:  # even row - uses second color
                    if second_used + row <= second_color_count:
                        second_used += row
                        height = row
                    else:
                        break
            
            return height
        
        # Try both patterns: red first, then blue first
        pattern1 = max_height_with_first_color(red, blue)
        pattern2 = max_height_with_first_color(blue, red)
        
        return max(pattern1, pattern2)