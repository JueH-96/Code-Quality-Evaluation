class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort()
        total_sum = 0
        for i, height in enumerate(maximumHeight):
            if total_sum < height * (i + 1):
                return -1
            total_sum += height * (i + 1)
        return total_sum