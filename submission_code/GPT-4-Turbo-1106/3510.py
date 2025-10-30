class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort(reverse=True)
        max_sum = 0
        current_height = float('inf')
        
        for height in maximumHeight:
            if current_height > height:
                current_height = height
            elif current_height > 1:
                current_height -= 1
            else:
                return -1
            max_sum += current_height
        
        return max_sum