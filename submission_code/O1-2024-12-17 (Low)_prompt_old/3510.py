class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort in descending order to greedily pick the largest possible distinct heights
        maximumHeight.sort(reverse=True)
        
        total_sum = 0
        # This variable will track what the next height must be below
        # so that each tower has a strictly smaller height than the previous one.
        current_max_allowed = float('inf')
        
        for h in maximumHeight:
            # Choose the largest possible distinct height not exceeding h and current_max_allowed - 1
            # (current_max_allowed - 1 is the maximum we can assign to this tower to ensure distinctness)
            assignable_height = min(h, current_max_allowed - 1)
            
            # If we can't assign a positive integer, it's impossible to find a valid arrangement
            if assignable_height < 1:
                return -1
            
            total_sum += assignable_height
            current_max_allowed = assignable_height  # Update the bound for the next tower
        
        return total_sum