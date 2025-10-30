import math

class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        total = red + blue
        h_max = int((math.sqrt(1 + 8 * total) - 1) // 2)
        
        for h in range(h_max, 0, -1):
            # Case 1: Starting with red
            k_red = (h + 1) // 2
            k_blue = h // 2
            red_used = k_red ** 2
            blue_used = k_blue * (k_blue + 1)
            if red >= red_used and blue >= blue_used:
                return h
            
            # Case 2: Starting with blue
            k_blue_case2 = (h + 1) // 2
            k_red_case2 = h // 2
            blue_used_case2 = k_blue_case2 ** 2
            red_used_case2 = k_red_case2 * (k_red_case2 + 1)
            if blue >= blue_used_case2 and red >= red_used_case2:
                return h
        
        return 0  # This return is theoretically unreachable as h starts from 1