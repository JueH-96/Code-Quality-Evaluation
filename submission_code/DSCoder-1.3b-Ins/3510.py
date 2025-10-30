class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort()
        totalSum = 0
        for i in range(len(maximumHeight)):
            totalSum += maximumHeight[i]
            maximumHeight[i] = maximumHeight[i] + 1
        return totalSum if totalSum > 0 else -1