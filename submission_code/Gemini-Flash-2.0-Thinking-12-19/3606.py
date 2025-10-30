class Solution:
    def minElement(self, nums: List[int]) -> int:
        digit_sums = []
        for num in nums:
            digit_sum = 0
            temp_num = num
            while temp_num > 0:
                digit_sum += temp_num % 10
                temp_num //= 10
            digit_sums.append(digit_sum)
        return min(digit_sums)