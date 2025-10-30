from typing import List
from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Map each value to the list of its indices in nums
        pos = defaultdict(list)
        for i, v in enumerate(nums):
            pos[v].append(i)

        ans = 0
        # For each distinct value, use a sliding window on its positions
        for v, idxs in pos.items():
            l = 0
            # idxs is a sorted list of positions where nums == v
            for r in range(len(idxs)):
                # Compute deletions needed to make nums[idxs[l]..idxs[r]] all equal to v
                # Total span length = idxs[r] - idxs[l] + 1
                # But we have (r-l+1) actual v's, so need to delete the rest
                while l <= r and (idxs[r] - idxs[l] + 1) - (r - l + 1) > k:
                    l += 1
                # After ensuring deletions <= k, window size is (r-l+1)
                ans = max(ans, r - l + 1)

        return ans