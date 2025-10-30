from typing import List

class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        # The difference between the left and right sums is 2*left_sum - total_sum.
        # 2*left_sum is always even, so the parity of the difference depends only on total_sum.
        # If total_sum is even then every split gives an even difference.
        # If total_sum is odd then every split gives an odd difference.
        return len(nums) - 1 if total_sum % 2 == 0 else 0