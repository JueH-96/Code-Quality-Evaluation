class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        diff = [target[i] - nums[i] for i in range(n)]
        operations = 0
        current = 0
        for d in diff:
            if d != current:
                operations += abs(d - current)
                current = d
        return operations