from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            left = (i - 1) % n
            right = (i + 1) % n
            if colors[i] != colors[left] and colors[i] != colors[right]:
                count += 1
        return count