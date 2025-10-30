from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        # Use 64-bit rolling hash (mod 2^64 for speed, low collision risk here).
        B = 1315423911  # random odd base
        h = [0] * (n + 1)
        p = [1] * (n + 1)
        for i in range(n):
            h[i+1] = (h[i] * B + (nums[i] + 1)) & 0xFFFFFFFFFFFFFFFF
            p[i+1] = (p[i] * B) & 0xFFFFFFFFFFFFFFFF

        def get_hash(l: int, r: int) -> int:
            """Return hash of nums[l..r] inclusive."""
            return (h[r+1] - (h[l] * p[r - l + 1] & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF

        res1 = 0  # count splits satisfying nums1 is prefix of nums2
        res2 = 0  # count splits satisfying nums2 is prefix of nums3
        res_int = 0  # count splits satisfying both

        # Precompute for each i whether prefix1 holds (nums[0:i]==nums[i:2i])
        pref1 = [False] * (n+1)
        for i in range(1, n//2 + 1):
            # we need positions [0..i-1] and [i..2i-1]
            if get_hash(0, i-1) == get_hash(i, 2*i-1):
                pref1[i] = True
                # number of valid j is j from 2i to n-1 inclusive => count = n - 2i
                res1 += n - 2*i

        # Now handle prefix2: for each i, j from i+1 to floor((n+i)//2)
        for i in range(1, n-1):
            maxj = (n + i) // 2
            if maxj <= i:
                continue
            for j in range(i+1, maxj+1):
                length = j - i
                # compare nums[i..j-1] with nums[j..j+length-1]
                if get_hash(i, j-1) == get_hash(j, j+length-1):
                    res2 += 1
                    # if prefix1 also holds, and j>=2i, it's in intersection
                    if pref1[i] and j >= 2*i:
                        res_int += 1

        return res1 + res2 - res_int