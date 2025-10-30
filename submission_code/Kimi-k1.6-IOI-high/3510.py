class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort(reverse=True)
        total = 0
        prev = float('inf')
        for h in maximumHeight:
            current = min(h, prev - 1)
            if current < 1:
                return -1
            total += current
            prev = current
        return total