class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        
        # Calculate the difference array
        diff = [target[i] - nums[i] for i in range(n)]
        
        operations = abs(diff[0])
        
        for i in range(1, n):
            # If signs are different, we need to add the full absolute value
            if diff[i] * diff[i-1] < 0:
                operations += abs(diff[i])
            # If signs are the same, we only add if absolute value increased
            elif abs(diff[i]) > abs(diff[i-1]):
                operations += abs(diff[i]) - abs(diff[i-1])
        
        return operations