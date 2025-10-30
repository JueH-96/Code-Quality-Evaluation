class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in descending order
        maximumHeight.sort(reverse=True)
        
        # Initialize a set to keep track of assigned heights
        assigned_heights = set()
        total_sum = 0
        
        for height in maximumHeight:
            # Start from the maximum allowed height and go downwards
            while height > 0 and height in assigned_heights:
                height -= 1
            
            # If we found a valid height, add it to the total sum and the set
            if height > 0:
                assigned_heights.add(height)
                total_sum += height
            else:
                # If we can't assign a valid height, return -1
                return -1
        
        return total_sum