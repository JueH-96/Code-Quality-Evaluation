from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_val = 0
        # Try all triplets (i, j, k) with i < j < k
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                diff = nums[i] - nums[j]
                # If diff <= 0, no need to check k because (diff * nums[k]) <= 0
                if diff <= 0:
                    continue
                for k in range(j + 1, n):
                    val = diff * nums[k]
                    if val > max_val:
                        max_val = val
        return max_val