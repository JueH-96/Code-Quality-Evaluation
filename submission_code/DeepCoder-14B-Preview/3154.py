import math

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        max_val = -math.inf
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    current = (nums[i] - nums[j]) * nums[k]
                    if current > max_val:
                        max_val = current
        return max_val if max_val >= 0 else 0