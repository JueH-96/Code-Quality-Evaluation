class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort maximum heights in descending order
        maximumHeight.sort(reverse=True)
        
        total_sum = 0
        next_available_height = 10**9 + 1  # Start with a value larger than any possible height
        
        for i in range(len(maximumHeight)):
            # The height is the minimum of the tower's maximum allowed height
            # and the next available height (to ensure distinctness)
            height = min(maximumHeight[i], next_available_height)
            
            if height <= 0:
                return -1  # No positive height available
            
            total_sum += height
            next_available_height = height - 1  # Next height must be distinct
        
        return total_sum