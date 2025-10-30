class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Determine the longest sequential prefix
        length = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                length += 1
            else:
                break
        
        # Calculate the sum of the longest sequential prefix
        sum_val = sum(nums[:length])
        
        # Use a set for O(1) lookups
        num_set = set(nums)
        
        # Find the smallest x >= sum_val not in the set
        x = sum_val
        while True:
            if x not in num_set:
                return x
            x += 1