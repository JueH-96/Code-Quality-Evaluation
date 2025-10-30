from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        # Total sum of all elements
        total = sum(nums)
        # Count frequencies of each number
        freq = {}
        for v in nums:
            freq[v] = freq.get(v, 0) + 1

        # We'll track the maximum candidate outlier
        # Values in nums lie in [-1000,1000], so initialize below that
        ans = -10**9

        # For each possible "sum" value v, compute the candidate outlier:
        # outlier = total - 2*v
        # and check validity based on frequencies.
        for v, cnt in freq.items():
            cand = total - 2 * v
            # The candidate outlier must exist in the array
            if cand not in freq:
                continue
            # If cand equals v, we need at least two occurrences to choose
            # one as the "sum" index and one as the "outlier" index
            if cand == v and freq[v] < 2:
                continue
            # Update the maximum outlier found
            ans = max(ans, cand)

        return ans