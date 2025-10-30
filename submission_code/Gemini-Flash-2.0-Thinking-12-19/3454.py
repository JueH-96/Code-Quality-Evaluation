class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        diff_array = [target[i] - nums[i] for i in range(len(nums))]
        operations = 0
        current_diff = 0
        for diff in diff_array:
            op = diff - current_diff
            operations += abs(op)
            current_diff += op
        return operations