class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the tower limits in ascending order
        maximumHeight.sort()
        
        n = len(maximumHeight)
        
        # Start from the largest tower limit and assign the largest possible height
        current_height = maximumHeight[-1]
        total_sum = current_height
        
        # Traverse from the second-largest downward, assigning strictly smaller distinct heights
        for i in range(n - 2, -1, -1):
            # Assign a height that is at most one less than the previously assigned height
            # but also cannot exceed the maximum allowed for this tower
            next_height = min(maximumHeight[i], current_height - 1)
            if next_height <= 0:
                return -1
            total_sum += next_height
            current_height = next_height
        
        return total_sum