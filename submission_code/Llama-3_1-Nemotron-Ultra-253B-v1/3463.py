from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            left = colors[i]
            middle = colors[(i + 1) % n]
            right = colors[(i + 2) % n]
            if middle != left and middle != right:
                count += 1
        return count