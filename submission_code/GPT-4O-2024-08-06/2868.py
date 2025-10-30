class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = 0
        left = 0
        
        for right in range(n):
            while left < right and max(nums[left:right+1]) - min(nums[left:right+1]) > 2:
                left += 1
            total_subarrays += (right - left + 1)
        
        return total_subarrays