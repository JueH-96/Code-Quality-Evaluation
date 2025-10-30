class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def compute(start_with_red, r, b):
            height = 0
            current_r = r
            current_b = b
            current_color = start_with_red  # True for red, False for blue
            row = 1
            while True:
                if current_color:
                    if current_r >= row:
                        current_r -= row
                        height += 1
                    else:
                        break
                else:
                    if current_b >= row:
                        current_b -= row
                        height += 1
                    else:
                        break
                current_color = not current_color
                row += 1
            return height
        
        height_red_start = compute(True, red, blue)
        height_blue_start = compute(False, red, blue)
        return max(height_red_start, height_blue_start)