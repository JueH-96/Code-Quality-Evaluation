from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_value = 0
        max_diff = 0
        max_num = 0

        for k in range(n - 1, 1, -1):
            for j in range(k - 1, 0, -1):
                for i in range(j - 1, -1, -1):
                    value = (nums[i] - nums[j]) * nums[k]
                    if value > max_value:
                        max_value = value

        return max_value if max_value > 0 else 0