from typing import List

class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        max_sum = -1
        max_area = 0
        for l, w in dimensions:
            sum_sq = l ** 2 + w ** 2
            area = l * w
            if sum_sq > max_sum:
                max_sum = sum_sq
                max_area = area
            elif sum_sq == max_sum:
                if area > max_area:
                    max_area = area
        return max_area