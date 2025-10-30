class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort()
        total = 0
        h_prev = float('inf')
        for h in reversed(maximumHeight):
            h_i = min(h, h_prev - 1)
            if h_i <= 0:
                return -1
            total += h_i
            h_prev = h_i
        return total