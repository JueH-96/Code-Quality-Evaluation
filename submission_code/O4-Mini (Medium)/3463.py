from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            left = colors[(i - 1) % n]
            mid = colors[i]
            right = colors[(i + 1) % n]
            if mid != left and mid != right:
                count += 1
        return count