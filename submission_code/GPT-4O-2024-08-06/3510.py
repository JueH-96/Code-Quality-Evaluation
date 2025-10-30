class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in descending order
        maximumHeight.sort(reverse=True)
        
        # Initialize the total sum and the current height to assign
        total_sum = 0
        current_height = float('inf')
        
        for max_height in maximumHeight:
            # Assign the maximum possible height that is less than or equal to max_height
            if current_height > max_height:
                current_height = max_height
            else:
                current_height -= 1
            
            # If the current height becomes zero or negative, it's impossible to assign unique heights
            if current_height <= 0:
                return -1
            
            # Add the current height to the total sum
            total_sum += current_height
        
        return total_sum