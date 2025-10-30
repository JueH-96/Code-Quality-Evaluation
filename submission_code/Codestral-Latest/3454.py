class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        # Calculate the difference between each corresponding element in nums and target
        diffs = [abs(nums[i] - target[i]) for i in range(len(nums))]

        # The minimum number of operations is the sum of these differences
        return sum(diffs)