class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        position = 0
        count = 0
        
        for move in nums:
            position += move
            if position == 0:
                count += 1
        
        return count