from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            # Considering the 3 contiguous tiles from i, i+1, i+2 (circularly)
            left = colors[i % n]
            mid = colors[(i + 1) % n]
            right = colors[(i + 2) % n]
            if mid != left and mid != right:
                count += 1
        return count