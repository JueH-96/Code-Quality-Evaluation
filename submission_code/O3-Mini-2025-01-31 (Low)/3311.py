class Solution:
    def returnToBoundaryCount(self, nums: list[int]) -> int:
        count = 0
        position = 0
        
        for move in nums:
            # Update the position based on move
            position += move
            # Count if the ant returns exactly to position 0 (the boundary) after the move
            if position == 0:
                count += 1
        
        return count