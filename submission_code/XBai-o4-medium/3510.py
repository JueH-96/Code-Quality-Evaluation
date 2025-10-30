class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort(reverse=True)
        total = 0
        prev = float('inf')
        for height in maximumHeight:
            current_max = min(height, prev - 1)
            if current_max < 1:
                return -1
            total += current_max
            prev = current_max
        return total