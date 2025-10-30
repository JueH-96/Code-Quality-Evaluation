class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        idx = 2
        count = 0
        
        while idx < len(nums):
            if nums[idx - 2] + nums[idx] == nums[idx - 1] / 2:
                count += 1
            idx += 1
            
        return count