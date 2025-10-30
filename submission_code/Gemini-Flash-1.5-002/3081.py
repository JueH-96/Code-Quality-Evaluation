class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        if n % 2 == 1:
            return 1
        else:
            return 0