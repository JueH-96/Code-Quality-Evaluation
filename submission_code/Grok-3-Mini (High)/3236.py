class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Find the length of the longest sequential prefix
        prefix_len = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                prefix_len += 1
            else:
                break
        
        # Calculate the sum of the sequential prefix
        prefix_sum = sum(nums[:prefix_len])
        
        # Create a set of all numbers in nums for O(1) lookups
        num_set = set(nums)
        
        # Find the smallest missing integer >= prefix_sum
        x = prefix_sum
        while True:
            if x not in num_set:
                return x
            x += 1