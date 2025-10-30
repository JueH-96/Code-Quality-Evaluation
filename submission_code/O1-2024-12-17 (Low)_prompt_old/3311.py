class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        position = 0
        count = 0
        for step in nums:
            position += step
            if position == 0:
                count += 1
        return count