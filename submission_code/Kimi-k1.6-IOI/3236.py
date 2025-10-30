class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Find the longest sequential prefix starting from the first element
        current_length = 1
        n = len(nums)
        for i in range(1, n):
            if nums[i] == nums[i-1] + 1:
                current_length += 1
            else:
                break
        
        # Calculate the sum of the longest sequential prefix
        sum_target = sum(nums[:current_length])
        
        # Create a set of the numbers for O(1) lookups
        num_set = set(nums)
        
        # Find the smallest integer >= sum_target that is missing
        x = sum_target
        while True:
            if x not in num_set:
                return x
            x += 1