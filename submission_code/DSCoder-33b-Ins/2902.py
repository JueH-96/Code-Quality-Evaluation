class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_sum = -1
        digit_dict = {}

        for num in nums:
            max_digit = max(str(num))
            if max_digit not in digit_dict:
                digit_dict[max_digit] = num
            else:
                max_sum = max(max_sum, num + digit_dict[max_digit])
                digit_dict[max_digit] = max(num, digit_dict[max_digit])

        return max_sum