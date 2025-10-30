class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        count = 0
        boundary = 0
        for num in nums:
            boundary += num
            if boundary < 0:
                count += 1
                boundary = 0
        return count