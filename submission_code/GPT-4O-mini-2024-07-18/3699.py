from typing import List

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)

        # Iterate through all possible combinations of (p, q, r, s)
        for p in range(n - 3):
            for q in range(p + 2, n - 1):
                for r in range(q + 2, n):
                    for s in range(r + 2, n):
                        if nums[p] * nums[r] == nums[q] * nums[s]:
                            count += 1

        return count