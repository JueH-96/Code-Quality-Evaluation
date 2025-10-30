class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in descending order
        maximumHeight.sort(reverse=True)
        
        # Initialize the last assigned height to a value larger than any possible height
        last_assigned = float('inf')
        total_sum = 0
        
        for height in maximumHeight:
            # Assign the current height as the minimum of its maximum height and one less than the last assigned height
            assign_height = min(height, last_assigned - 1)
            if assign_height < 1:
                return -1
            total_sum += assign_height
            last_assigned = assign_height
        
        return total_sum