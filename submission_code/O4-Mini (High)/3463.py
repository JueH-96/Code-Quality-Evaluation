from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            prev_color = colors[(i - 1) % n]
            next_color = colors[(i + 1) % n]
            # middle tile must differ from both neighbors
            if colors[i] != prev_color and colors[i] != next_color:
                count += 1
        return count