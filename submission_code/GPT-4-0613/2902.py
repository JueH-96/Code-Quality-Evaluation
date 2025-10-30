class Solution:
    def maxSum(self, nums):
        max_digit_dict = {}
        for num in nums:
            max_digit = max(int(digit) for digit in str(num))
            if max_digit in max_digit_dict:
                max_digit_dict[max_digit] = max(max_digit_dict[max_digit], num)
            else:
                max_digit_dict[max_digit] = num

        max_sum = -1
        for num in nums:
            max_digit = max(int(digit) for digit in str(num))
            if max_digit_dict[max_digit] != num:
                max_sum = max(max_sum, num + max_digit_dict[max_digit])

        return max_sum