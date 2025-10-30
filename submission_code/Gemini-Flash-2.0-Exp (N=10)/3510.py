class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        if n > len(set(maximumHeight)):
            return -1
        
        maximumHeight.sort(reverse=True)
        
        total_sum = 0
        
        assigned_heights = set()
        
        for max_height in maximumHeight:
            
            height = max_height
            while height > 0 and height in assigned_heights:
                height -= 1
            
            if height == 0:
                return -1
            
            assigned_heights.add(height)
            total_sum += height
            
        return total_sum