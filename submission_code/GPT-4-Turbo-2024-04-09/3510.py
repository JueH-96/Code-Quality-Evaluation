class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort()
        n = len(maximumHeight)
        current_height = 0
        total_sum = 0
        
        for i in range(n):
            if current_height >= maximumHeight[i]:
                return -1
            current_height = max(current_height + 1, maximumHeight[i] - (n - i - 1))
            total_sum += current_height
        
        return total_sum