class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        if n > len(set(maximumHeight)):
            return -1
        
        heights = sorted(maximumHeight)
        total_sum = 0
        for i in range(n):
            total_sum += heights[i]
        
        return total_sum