class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0  # Edge case, though n >=1 as per constraints
        
        left = 0
        right = n - 1
        pairs = 0
        
        while left < right:
            if nums[left] < nums[right]:
                pairs += 1
                left += 1
                right -= 1
            else:
                right -= 1
                
        return n - 2 * pairs