class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        total = red + blue
        if total == 0:
            return 0
        # Calculate the maximum possible height
        h_max = int((( (1 + 8 * total) ** 0.5 ) - 1) // 2)
        for h in range(h_max, 0, -1):
            # Try starting with red
            r_red = (h + 1) // 2
            sum_red = r_red ** 2
            sum_blue = (h // 2) * (h // 2 + 1)
            if sum_red <= red and sum_blue <= blue:
                return h
            # Try starting with blue
            r_blue = (h + 1) // 2
            sum_blue_start = r_blue ** 2
            sum_red_start = (h // 2) * (h // 2 + 1)
            if sum_blue_start <= blue and sum_red_start <= red:
                return h
        return 0  # This line is theoretically unreachable given constraints