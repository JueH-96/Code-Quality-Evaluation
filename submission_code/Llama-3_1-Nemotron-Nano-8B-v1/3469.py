class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        for h in range(200, 0, -1):
            # Check starting with red
            red_rows = (h + 1) // 2
            blue_rows = h // 2
            sum_red = red_rows ** 2
            sum_blue = blue_rows * (blue_rows + 1)
            if sum_red <= red and sum_blue <= blue:
                return h
            # Check starting with blue
            sum_blue_start = ((h + 1) // 2) ** 2
            sum_red_start = (h // 2) * (h // 2 + 1)
            if sum_blue_start <= blue and sum_red_start <= red:
                return h
        return 0  # This line is theoretically unreachable