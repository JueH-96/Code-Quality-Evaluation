class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        if n % 2 == 0:
            return 0
        else:
            return 1