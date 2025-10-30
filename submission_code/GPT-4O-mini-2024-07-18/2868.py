class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        left = 0
        
        for right in range(n):
            while max(nums[left:right + 1]) - min(nums[left:right + 1]) > 2:
                left += 1
            count += right - left + 1
        
        return count