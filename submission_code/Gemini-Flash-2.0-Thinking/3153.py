class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        counts = [0] * 32  # Assuming max value of nums[i] fits in 32 bits
        for num in nums:
            for i in range(32):
                if (num >> i) & 1:
                    counts[i] += 1

        result_nums = [0] * k
        for i in range(31, -1, -1):
            ones_at_bit = counts[i]
            for j in range(min(k, ones_at_bit)):
                result_nums[j] |= (1 << i)

        mod = 10**9 + 7
        max_sum_squares = 0
        for num in result_nums:
            max_sum_squares = (max_sum_squares + num * num) % mod

        return max_sum_squares