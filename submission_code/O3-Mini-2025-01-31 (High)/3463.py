from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            # Using modular arithmetic to consider the circle:
            # colors[i-1] is the left neighbor (negative index handles wrap-around)
            # colors[(i+1) % n] is the right neighbor
            if colors[i] != colors[i - 1] and colors[i] != colors[(i + 1) % n]:
                count += 1
        return count