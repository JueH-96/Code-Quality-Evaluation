from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        """
        Maximum sub-array sum whose length is a multiple of k.

        Idea:
        A sub-array nums[i … j] (inclusive) has length L = j-i+1.
        L is divisible by k  ⇔  (j+1) and i share the same remainder mod k.
        Using prefix sums S[t] = sum(nums[0:t]), the sub-array sum is
            S[j+1] – S[i].
        For every remainder r (0 … k-1) we keep the smallest prefix value
        encountered so far with that remainder.  While scanning the array we
        can therefore obtain in O(1) the best candidate that ends at the
        current position.

        Complexity: O(n) time,  O(k) memory.
        """
        INF = float('inf')
        # min_prefix[r] = minimum prefix sum seen so far whose index % k == r
        min_prefix = [INF] * k
        min_prefix[0] = 0          # prefix sum at index 0
        best = -10**19             # smaller than the minimum possible answer
        prefix = 0

        for idx, val in enumerate(nums, 1):   # idx is 1-based prefix index
            prefix += val
            r = idx % k

            if min_prefix[r] != INF:
                best = max(best, prefix - min_prefix[r])

            if prefix < min_prefix[r]:
                min_prefix[r] = prefix

        return best