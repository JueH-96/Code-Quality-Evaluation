class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_sum = -1
        n = len(nums)

        def get_max_digit(num):
            max_digit = 0
            for digit in str(num):
                max_digit = max(max_digit, int(digit))
            return max_digit

        for i in range(n):
            for j in range(i + 1, n):
                max_digit_i = get_max_digit(nums[i])
                max_digit_j = get_max_digit(nums[j])

                if max_digit_i == max_digit_j:
                    current_sum = nums[i] + nums[j]
                    max_sum = max(max_sum, current_sum)

        return max_sum