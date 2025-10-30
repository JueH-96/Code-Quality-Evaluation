class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_sum = -1
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                max_digit_i = 0
                temp = nums[i]
                while temp > 0:
                    max_digit_i = max(max_digit_i, temp % 10)
                    temp //= 10
                max_digit_j = 0
                temp = nums[j]
                while temp > 0:
                    max_digit_j = max(max_digit_j, temp % 10)
                    temp //= 10
                if max_digit_i == max_digit_j:
                    max_sum = max(max_sum, nums[i] + nums[j])
        return max_sum