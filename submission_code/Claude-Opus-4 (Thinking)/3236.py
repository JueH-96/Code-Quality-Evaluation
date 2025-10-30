class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Find the longest sequential prefix
        prefix_length = 1  # At least the first element
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                prefix_length += 1
            else:
                break
        
        # Calculate the sum of the sequential prefix
        prefix_sum = sum(nums[:prefix_length])
        
        # Convert nums to a set for O(1) lookup
        nums_set = set(nums)
        
        # Find the smallest missing integer >= prefix_sum
        x = prefix_sum
        while x in nums_set:
            x += 1
        
        return x