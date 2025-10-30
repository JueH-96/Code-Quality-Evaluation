from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # prefix_or[i] will be OR of nums[0..i-1]
        prefix_or = [0] * n
        cur = 0
        for i in range(n):
            prefix_or[i] = cur
            cur |= nums[i]

        # suffix_or[i] will be OR of nums[i+1..n-1]
        suffix_or = [0] * n
        cur = 0
        for i in range(n - 1, -1, -1):
            suffix_or[i] = cur
            cur |= nums[i]

        ans = 0
        # try applying all k doublings to each index i
        for i in range(n):
            without_i = prefix_or[i] | suffix_or[i]
            with_i = nums[i] << k
            combined = without_i | with_i
            if combined > ans:
                ans = combined

        return ans