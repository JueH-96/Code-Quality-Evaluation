class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Find the longest sequential prefix
        longest_prefix_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                longest_prefix_sum += nums[i]
            else:
                break
        
        # Find the smallest integer x missing from nums such that x >= longest_prefix_sum
        x = longest_prefix_sum
        while x in nums:
            x += 1
        
        return x