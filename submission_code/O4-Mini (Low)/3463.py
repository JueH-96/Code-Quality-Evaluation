from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        for i in range(n):
            left = (i - 1) % n
            right = (i + 1) % n
            # check if left and right are the same color,
            # and the middle is different
            if colors[left] == colors[right] and colors[i] != colors[left]:
                count += 1
        return count