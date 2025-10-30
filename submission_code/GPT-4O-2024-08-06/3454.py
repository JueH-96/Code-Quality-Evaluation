class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        # Calculate the difference array between target and nums
        diff = [target[i] - nums[i] for i in range(len(nums))]
        
        # Initialize the number of operations
        operations = 0
        
        # Iterate through the difference array
        for i in range(len(diff)):
            if i == 0:
                # For the first element, we need to perform operations equal to its absolute value
                operations += abs(diff[i])
            else:
                # For subsequent elements, we need to perform operations equal to the absolute difference
                # between the current and previous elements in the difference array
                operations += abs(diff[i] - diff[i - 1])
        
        return operations