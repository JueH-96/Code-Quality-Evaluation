class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort in descending order to process towers with higher max heights first
        maximumHeight.sort(reverse=True)
        
        total_sum = 0
        last_assigned = None
        
        for max_h in maximumHeight:
            if last_assigned is None:
                assigned_height = max_h
            else:
                assigned_height = min(max_h, last_assigned - 1)
            
            # If assigned height is not positive, it's impossible
            if assigned_height <= 0:
                return -1
            
            total_sum += assigned_height
            last_assigned = assigned_height
        
        return total_sum