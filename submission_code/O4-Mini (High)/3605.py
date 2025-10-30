from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for t in nums:
            best = None
            # try all possible k such that 2^k <= t
            for k in range(t.bit_length() + 1):
                two_k = 1 << k
                if two_k > t:
                    break
                x = t - two_k
                if x < 0:
                    continue
                # bit k of x must be 0
                if ((x >> k) & 1) != 0:
                    continue
                # bits 0..k-1 of x must all be 1
                if k > 0:
                    mask = (1 << k) - 1
                    if (x & mask) != mask:
                        continue
                # valid candidate
                if best is None or x < best:
                    best = x
            ans.append(best if best is not None else -1)
        return ans