class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in descending order
        maximumHeight.sort(reverse=True)
        
        # Initialize the current height to be assigned
        current_height = maximumHeight[0]
        
        # Initialize the total sum
        total_sum = 0
        
        for max_height in maximumHeight:
            if current_height > max_height:
                current_height = max_height
            
            if current_height <= 0:
                return -1
            
            total_sum += current_height
            current_height -= 1
        
        return total_sum