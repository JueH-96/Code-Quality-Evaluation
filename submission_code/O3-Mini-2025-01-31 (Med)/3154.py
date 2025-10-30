from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_value = float('-inf')
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    triplet_value = (nums[i] - nums[j]) * nums[k]
                    if triplet_value > max_value:
                        max_value = triplet_value
        return max_value if max_value > 0 else 0