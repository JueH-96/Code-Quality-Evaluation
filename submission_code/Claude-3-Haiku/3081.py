class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        
        left, right = 0, n - 1
        while left < right:
            if nums[left] < nums[right]:
                left += 1
                right -= 1
            else:
                break
        
        return n - (right - left + 1)