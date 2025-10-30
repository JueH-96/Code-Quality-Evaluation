class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        sum_h = sum(horizontalCut)
        sum_v = sum(verticalCut)
        return sum_h * (m - 1) + sum_v * (n - 1)