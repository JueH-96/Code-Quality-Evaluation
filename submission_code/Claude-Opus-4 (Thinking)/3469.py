class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def check_height(first_color, second_color):
            height = 0
            first_used = second_used = 0
            
            while True:
                height += 1
                if height % 2 == 1:  # odd row uses first color
                    first_used += height
                    if first_used > first_color:
                        return height - 1
                else:  # even row uses second color
                    second_used += height
                    if second_used > second_color:
                        return height - 1
        
        # Try both starting colors and return the maximum
        return max(check_height(red, blue), check_height(blue, red))