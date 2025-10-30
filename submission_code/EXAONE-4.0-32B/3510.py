class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        arr = sorted(maximumHeight, reverse=True)
        last = float('inf')
        total = 0
        for x in arr:
            candidate = min(last - 1, x)
            if candidate < 1:
                return -1
            total += candidate
            last = candidate
        return total