class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # We can try both possibilities:
        # 1) Starting with a red row first
        # 2) Starting with a blue row first
        # Then take the maximum result.

        def build_triangle(r, b, start_color):
            height = 0
            current_color = start_color
            size = 1
            while True:
                if current_color == 'R':
                    if r >= size:
                        r -= size
                        height += 1
                        current_color = 'B'
                        size += 1
                    else:
                        break
                else:  # current_color == 'B'
                    if b >= size:
                        b -= size
                        height += 1
                        current_color = 'R'
                        size += 1
                    else:
                        break
            return height

        # Option 1: start with red
        option1 = build_triangle(red, blue, 'R')
        # Option 2: start with blue
        option2 = build_triangle(red, blue, 'B')

        return max(option1, option2)