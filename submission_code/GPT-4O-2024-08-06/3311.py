class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        position = 0
        boundary_count = 0
        
        for move in nums:
            position += move
            if position == 0:
                boundary_count += 1
        
        return boundary_count