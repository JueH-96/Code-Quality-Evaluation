class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort in descending order to prioritize higher maximum heights
        maximumHeight.sort(reverse=True)
        
        total_sum = 0
        last_assigned = float('inf')
        
        for max_h in maximumHeight:
            # Assign the highest valid height
            height = min(max_h, last_assigned - 1)
            
            # Check if we can assign a positive height
            if height <= 0:
                return -1
            
            total_sum += height
            last_assigned = height
        
        return total_sum