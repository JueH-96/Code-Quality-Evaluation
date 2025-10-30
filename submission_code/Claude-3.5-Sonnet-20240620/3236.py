class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Find the longest sequential prefix
        prefix_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1] + 1:
                break
            prefix_sum += nums[i]
        
        # Create a set of all numbers in nums for efficient lookup
        num_set = set(nums)
        
        # Find the smallest missing integer >= prefix_sum
        x = prefix_sum
        while x in num_set:
            x += 1
        
        return x