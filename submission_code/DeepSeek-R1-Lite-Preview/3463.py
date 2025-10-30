from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        # Extend the colors array to handle circular wrapping
        extended_colors = colors + colors[:2]
        count = 0
        # Iterate through the original number of positions
        for i in range(n):
            group = extended_colors[i:i+3]
            if group == [0, 1, 0] or group == [1, 0, 1]:
                count += 1
        return count