class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        boundary = 0
        count = 0
        for num in nums:
            boundary += num
            if boundary == 0:
                count += 1
        return count