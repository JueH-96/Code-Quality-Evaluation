class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def getMaxDigit(num):
            max_digit = 0
            for digit in str(num):
                max_digit = max(max_digit, int(digit))
            return max_digit

        max_sum = -1
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                max_digit1 = getMaxDigit(nums[i])
                max_digit2 = getMaxDigit(nums[j])
                if max_digit1 == max_digit2:
                    current_sum = nums[i] + nums[j]
                    max_sum = max(max_sum, current_sum)
        return max_sum