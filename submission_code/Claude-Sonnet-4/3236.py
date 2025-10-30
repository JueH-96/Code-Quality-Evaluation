class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Find the longest sequential prefix
        prefix_sum = nums[0]
        i = 1
        while i < len(nums) and nums[i] == nums[i-1] + 1:
            prefix_sum += nums[i]
            i += 1
        
        # Convert nums to set for O(1) lookup
        nums_set = set(nums)
        
        # Find the smallest missing integer >= prefix_sum
        x = prefix_sum
        while x in nums_set:
            x += 1
        
        return x