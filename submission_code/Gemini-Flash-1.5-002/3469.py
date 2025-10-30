class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        max_height = 0
        for start_color in [0, 1]:  # 0 for red, 1 for blue
            red_count = red
            blue_count = blue
            height = 0
            current_color = start_color
            row_length = 1
            while True:
                if current_color == 0:
                    if red_count >= row_length:
                        red_count -= row_length
                        height += 1
                        current_color = 1
                    else:
                        break
                else:
                    if blue_count >= row_length:
                        blue_count -= row_length
                        height += 1
                        current_color = 0
                    else:
                        break
                row_length += 1

            max_height = max(max_height, height)
        return max_height