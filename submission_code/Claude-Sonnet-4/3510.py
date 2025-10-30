class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort in descending order to assign highest possible heights first
        maximumHeight.sort(reverse=True)
        
        total_sum = 0
        current_height = float('inf')  # Start with infinity so first tower gets its max height
        
        for max_h in maximumHeight:
            # Assign the minimum of max_h and (current_height - 1)
            # This ensures uniqueness and maximizes the height
            assigned_height = min(max_h, current_height - 1)
            
            # If assigned height is not positive, it's impossible
            if assigned_height <= 0:
                return -1
            
            total_sum += assigned_height
            current_height = assigned_height
        
        return total_sum