class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        # Compute the difference array
        diff = [target[i] - nums[i] for i in range(len(nums))]
        
        # Initialize operations with the absolute value of the first difference
        operations = abs(diff[0])
        
        # Add the absolute difference between consecutive elements in the difference array
        for i in range(1, len(diff)):
            operations += abs(diff[i] - diff[i-1])
        
        return operations