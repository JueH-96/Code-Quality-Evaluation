class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        position = 0
        count = 0
        for num in nums:
            position += num
            if position == 0:
                count += 1
            position = max(min(position, 0), -1 * position)
        return count