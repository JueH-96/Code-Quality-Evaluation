class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        max_h = 0
        for h in range(1, 21):  # h can be up to 20 based on constraints
            ceil_h2 = (h + 1) // 2
            floor_h2 = h // 2
            # Starting with red
            red_required_start_red = ceil_h2 ** 2
            blue_required_start_red = floor_h2 * (floor_h2 + 1)
            # Starting with blue
            blue_required_start_blue = ceil_h2 ** 2
            red_required_start_blue = floor_h2 * (floor_h2 + 1)
            
            if (red_required_start_red <= red and blue_required_start_red <= blue) or \
               (red_required_start_blue <= red and blue_required_start_blue <= blue):
                max_h = h
        return max_h