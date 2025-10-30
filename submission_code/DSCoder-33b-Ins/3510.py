class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort(reverse=True)
        total_sum = 0
        for i in range(len(maximumHeight)):
            if maximumHeight[i] < len(maximumHeight) - i:
                return -1
            total_sum += maximumHeight[i]
        return total_sum