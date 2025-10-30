class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        operations = 0
        diff = 0
        
        for i in range(n):
            current_diff = target[i] - nums[i]
            if current_diff != diff:
                operations += abs(current_diff - diff)
                diff = current_diff
        
        return operations