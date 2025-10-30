from collections import defaultdict
from bisect import bisect_left, insort

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        nums.sort()
        total = nums[0] ** 3
        prefix_sum = nums[0]
        for i in range(1, len(nums)):
            total += nums[i] ** 3
            total += nums[i] * nums[i] * prefix_sum
            prefix_sum *= 2
            prefix_sum += nums[i]
            prefix_sum %= 1000000007
            total %= 1000000007
        return total