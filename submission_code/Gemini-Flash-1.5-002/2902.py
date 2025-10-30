class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def max_digit(n):
            return max(int(digit) for digit in str(n))

        max_sum = -1
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if max_digit(nums[i]) == max_digit(nums[j]):
                    max_sum = max(max_sum, nums[i] + nums[j])
        return max_sum