class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)

        # Find all indices where nums[i] == 1
        ones_indices = [i for i in range(n) if nums[i] == 1]

        # If there are no 1s, return 0
        if not ones_indices:
            return 0

        # If there is only one 1, return 1
        if len(ones_indices) == 1:
            return 1

        # Calculate the number of ways to split the array
        result = 1
        for i in range(1, len(ones_indices)):
            # The number of ways to split between two consecutive 1s
            # is the number of zeros between them plus one
            zeros_between = ones_indices[i] - ones_indices[i - 1] - 1
            result = (result * (zeros_between + 1)) % MOD

        return result