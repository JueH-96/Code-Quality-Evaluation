class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def max_digit(n):
            max_d = 0
            while n > 0:
                max_d = max(max_d, n % 10)
                n //= 10
            return max_d

        max_sum = -1
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if max_digit(nums[i]) == max_digit(nums[j]):
                    max_sum = max(max_sum, nums[i] + nums[j])
        return max_sum