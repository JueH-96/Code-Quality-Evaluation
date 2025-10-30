class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        total_diff = 0
        num_digits = len(str(nums[0]))

        for digit_index in range(num_digits):
            digit_counts = {}
            for num in nums:
                digit_str = str(num)
                digit = digit_str[digit_index]
                digit_counts[digit] = digit_counts.get(digit, 0) + 1

            diff_at_position = 0
            for digit, count in digit_counts.items():
                diff_at_position += count * (n - count)

            total_diff += diff_at_position // 2

        return total_diff