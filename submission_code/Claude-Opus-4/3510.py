class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort in descending order to assign highest possible heights first
        maximumHeight.sort(reverse=True)
        
        total_sum = 0
        # Start with the highest possible height for the first tower
        current_height = maximumHeight[0]
        
        for max_h in maximumHeight:
            # Assign the minimum of the tower's max height and the current available height
            assigned_height = min(max_h, current_height)
            
            # If we need to assign 0 or negative height, it's impossible
            if assigned_height <= 0:
                return -1
            
            total_sum += assigned_height
            # Next available height is one less than what we just assigned
            current_height = assigned_height - 1
        
        return total_sum