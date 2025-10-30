from collections import defaultdict
from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # freq[r] will count how many prefix sums (modulo `modulo`) have the value r
        freq = defaultdict(int)
        # empty prefix (sum of zero elements) has sum = 0
        freq[0] = 1

        prefix = 0  # current prefix sum of "good" elements, taken mod `modulo`
        ans = 0

        for x in nums:
            # A[i] = 1 if x % modulo == k, else 0
            if x % modulo == k:
                # increment and reduce mod `modulo`
                prefix = (prefix + 1) % modulo

            # we want (prefix - prev_prefix) % modulo == k
            # => prev_prefix ≡ prefix - k  (mod modulo)
            needed = (prefix - k) % modulo
            ans += freq.get(needed, 0)

            # record this prefix for future subarrays
            freq[prefix] += 1

        return ans