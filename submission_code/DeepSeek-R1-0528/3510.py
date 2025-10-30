class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort()
        n = len(maximumHeight)
        for i in range(n):
            if maximumHeight[i] < i + 1:
                return -1
        current = float('inf')
        total_sum = 0
        for i in range(n - 1, -1, -1):
            current = min(maximumHeight[i], current - 1)
            total_sum += current
        return total_sum