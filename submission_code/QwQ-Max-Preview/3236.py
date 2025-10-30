class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        if not nums:
            return 0  # According to constraints, nums is non-empty
        
        # Find the longest sequential prefix
        max_length = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                max_length += 1
            else:
                break
        
        # Calculate the sum of the longest sequential prefix
        sum_val = sum(nums[:max_length])
        
        # Create a set for O(1) lookups
        num_set = set(nums)
        
        # Find the smallest x >= sum_val not in num_set
        current = sum_val
        while True:
            if current not in num_set:
                return current
            current += 1