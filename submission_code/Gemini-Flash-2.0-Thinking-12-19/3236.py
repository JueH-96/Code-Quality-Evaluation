class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 1

        longest_prefix_len = 1
        for i in range(1, n):
            if nums[i] == nums[i-1] + 1:
                longest_prefix_len += 1
            else:
                break

        prefix_sum = sum(nums[:longest_prefix_len])

        nums_set = set(nums)
        missing_int = prefix_sum
        while True:
            if missing_int not in nums_set:
                return missing_int
            missing_int += 1