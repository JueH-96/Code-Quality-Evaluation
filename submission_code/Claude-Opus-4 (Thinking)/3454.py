class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        diff = [target[i] - nums[i] for i in range(len(nums))]
        
        operations = 0
        current = 0
        
        for d in diff:
            if d * current < 0:  # Different sign (both non-zero)
                operations += abs(d)
            else:  # Same sign or at least one is 0
                operations += max(0, abs(d) - abs(current))
            current = d
        
        return operations