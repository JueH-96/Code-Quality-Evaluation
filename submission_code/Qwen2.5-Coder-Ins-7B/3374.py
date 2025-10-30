class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        left = 0
        
        for right in range(n):
            if right > 0 and nums[right] == nums[right - 1]:
                left = right
            count += right - left + 1
        
        return count