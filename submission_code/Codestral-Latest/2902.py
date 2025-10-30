class Solution:
    def maxSum(self, nums: List[int]) -> int:
        from collections import defaultdict

        def max_digit(num):
            return max(int(d) for d in str(num))

        max_sum = -1
        digit_dict = defaultdict(list)

        for num in nums:
            max_d = max_digit(num)
            digit_dict[max_d].append(num)

        for digit, values in digit_dict.items():
            if len(values) > 1:
                values.sort(reverse=True)
                max_sum = max(max_sum, values[0] + values[1])

        return max_sum