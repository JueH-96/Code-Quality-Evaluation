class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def get_max_digit(num):
            max_digit = 0
            for digit in str(num):
                max_digit = max(max_digit, int(digit))
            return max_digit

        max_sum_val = -1
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                max_digit1 = get_max_digit(nums[i])
                max_digit2 = get_max_digit(nums[j])
                if max_digit1 == max_digit2:
                    current_sum = nums[i] + nums[j]
                    max_sum_val = max(max_sum_val, current_sum)
        return max_sum_val