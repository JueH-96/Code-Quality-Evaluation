class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Find the longest sequential prefix
        longest_prefix_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                longest_prefix_sum += nums[i]
            else:
                break
        
        # Find the smallest integer missing from nums that is >= longest_prefix_sum
        x = longest_prefix_sum
        nums_set = set(nums)
        
        while x in nums_set:
            x += 1
        
        return x