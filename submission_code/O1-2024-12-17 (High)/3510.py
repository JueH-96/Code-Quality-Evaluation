class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the tower capacities in descending order
        maxHeights = sorted(maximumHeight, reverse=True)
        
        total_sum = 0
        # Keep track of the maximum height we can assign to the current tower
        # starting from a very large value
        current = float('inf')
        
        for c in maxHeights:
            # Assign the smallest possible distinct height (cannot exceed current or c)
            assigned = min(current, c)
            # If we cannot assign a positive height, return -1
            if assigned <= 0:
                return -1
            
            total_sum += assigned
            # Update the next available height to be one less than the current assignment
            current = assigned - 1
        
        return total_sum