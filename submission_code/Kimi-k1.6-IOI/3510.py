class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort(reverse=True)
        prev = float('inf')
        total = 0
        for h in maximumHeight:
            current = min(prev - 1, h)
            if current < 1:
                return -1
            total += current
            prev = current
        return total