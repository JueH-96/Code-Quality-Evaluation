class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        diff = [target[i] - nums[i] for i in range(len(nums))]
        
        operations = 0
        prev = 0
        
        for d in diff:
            if d * prev >= 0:  # Same sign or one is zero
                operations += abs(d - prev)
            else:  # Different signs
                operations += abs(d)
            prev = d
        
        return operations