class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Find the longest sequential prefix
        prefix_length = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                prefix_length += 1
            else:
                break
        
        # Calculate the sum of the prefix
        prefix_sum = sum(nums[:prefix_length])
        
        # Use a set for faster lookup
        nums_set = set(nums)
        
        # Find the smallest integer missing from the array that is >= prefix_sum
        x = prefix_sum
        while x in nums_set:
            x += 1
        
        return x