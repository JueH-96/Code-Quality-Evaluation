class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_sum = -1
        max_digit = [0]*10
        for num in nums:
            digit = int(max(str(num)))
            if max_digit[digit] > 0:
                max_sum = max(max_sum, max_digit[digit] + num)
                max_digit[digit] = max(max_digit[digit], num)
            else:
                max_digit[digit] = num
        return max_sum