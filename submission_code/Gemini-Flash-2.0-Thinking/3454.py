class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        diffs = [target[i] - nums[i] for i in range(n)]

        operations = abs(diffs[0])
        for i in range(1, n):
            operations += abs(diffs[i] - diffs[i-1])

        return operations