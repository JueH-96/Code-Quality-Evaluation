from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        sum_h = sum(horizontalCut)
        sum_v = sum(verticalCut)
        option1 = sum_v + sum_h * n
        option2 = sum_h + sum_v * m
        return min(option1, option2)