class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        total_digit_difference = 0
        for i in range(n):
            for j in range(i + 1, n):
                num1_str = str(nums[i])
                num2_str = str(nums[j])
                current_digit_difference = 0
                for k in range(len(num1_str)):
                    if num1_str[k] != num2_str[k]:
                        current_digit_difference += 1
                total_digit_difference += current_digit_difference
        return total_digit_difference