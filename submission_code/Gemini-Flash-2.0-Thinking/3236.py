class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 1  # Should not happen based on constraints

        longest_sequential_prefix_len = 1
        for i in range(1, n):
            if nums[i] == nums[i - 1] + 1:
                longest_sequential_prefix_len += 1
            else:
                break

        longest_sequential_prefix_sum = sum(nums[:longest_sequential_prefix_len])

        missing = longest_sequential_prefix_sum
        nums_set = set(nums)
        while True:
            if missing not in nums_set:
                return missing
            missing += 1