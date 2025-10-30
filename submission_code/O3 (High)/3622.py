from typing import List
from collections import Counter

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Original frequencies of every value in nums
        freq = Counter(nums)
        
        # If k == 0 we can only leave numbers unchanged
        # (adding 0 keeps them equal), so answer is the
        # original highest frequency.
        if k == 0 or numOperations == 0:
            return max(freq.values())
        
        # Range of every value reachable from some nums[i]
        lo = min(nums) - k
        hi = max(nums) + k
        
        size = hi - lo + 2          # +1 for inclusive hi, +1 for sentinel
        diff = [0] * size           # difference array for range-add counts
        
        # Each element contributes +1 to every T in [nums[i]-k , nums[i]+k]
        for x in nums:
            l = x - k - lo          # start index in diff
            r = x + k - lo          # end index in diff  (inclusive)
            diff[l]     += 1
            diff[r + 1] -= 1        # sentinel after the interval
        
        best = 0
        cur  = 0                    # running prefix sum = M[T]
        # Iterate over every possible T in [lo .. hi]
        for idx in range(size - 1):     # last cell is only the sentinel
            cur += diff[idx]
            if cur == 0:                # nothing convertible to this T
                continue
            T = lo + idx
            f0 = freq.get(T, 0)         # original occurrences of T
            # Maximum times we can make T appear:
            # can raise its count by at most numOperations,
            # but never beyond all convertible indices (cur)
            candidate = min(f0 + numOperations, cur)
            if candidate > best:
                best = candidate
        
        return best