class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort()
        total = 0
        prev = float('inf')
        for i in reversed(range(len(maximumHeight))):
            current = min(maximumHeight[i], prev - 1)
            if current < 1:
                return -1
            total += current
            prev = current
        return total