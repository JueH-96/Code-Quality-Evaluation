from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = float('-inf')

        # Calculate the prefix sums
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        # Use a dictionary to store the remainder and its index
        remainder_dict = {}

        for i in range(k, n + 1):
            remainder = prefix_sum[i] % k
            if remainder not in remainder_dict:
                remainder_dict[remainder] = i
            else:
                max_sum = max(max_sum, prefix_sum[i] - prefix_sum[remainder_dict[remainder]])

        return max_sum