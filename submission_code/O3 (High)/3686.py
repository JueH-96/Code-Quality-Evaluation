from typing import List

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0

        # -------- polynomial double hashing over the integer array --------
        BASE = 911382323          # base  <  MOD1 , MOD2
        MOD1 = 1_000_000_007
        MOD2 = 1_000_000_009

        pow1 = [1] * (n + 1)
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow1[i] = (pow1[i - 1] * BASE) % MOD1
            pow2[i] = (pow2[i - 1] * BASE) % MOD2

        pref1 = [0] * (n + 1)
        pref2 = [0] * (n + 1)
        for i, v in enumerate(nums):
            pref1[i + 1] = (pref1[i] * BASE + v + 1) % MOD1
            pref2[i + 1] = (pref2[i] * BASE + v + 1) % MOD2

        def sub_hash(pref, pw, l, r, mod):
            # hash of nums[l : r]  (0-based, r exclusive)
            return (pref[r] - pref[l] * pw[r - l]) % mod

        def equal_sub(a: int, b: int, length: int) -> bool:
            # are nums[a : a+length] and nums[b : b+length] identical ?
            return (
                sub_hash(pref1, pow1, a, a + length, MOD1)
                == sub_hash(pref1, pow1, b, b + length, MOD1)
                and
                sub_hash(pref2, pow2, a, a + length, MOD2)
                == sub_hash(pref2, pow2, b, b + length, MOD2)
            )

        # ----- pre–compute “is prefix” information for the first condition -----
        # prefix_match[i] : nums[0 : i] == nums[i : 2*i] (length i) ?
        prefix_match = [False] * n     # index 0 unused
        for i in range(1, n):
            # only need equality of the first i elements starting from 0 and i
            if i <= n - i:                 # there are at least i elements after i
                prefix_match[i] = equal_sub(0, i, i)

        # ----------------------- main enumeration -----------------------------
        ans = 0
        n_minus1 = n - 1
        for i in range(1, n_minus1):                      # first cut after position i-1
            len1 = i
            pref_ok = prefix_match[i]                     # nums1 is prefix of nums2?
            for j in range(i + 1, n):                     # second cut
                len2 = j - i
                len3 = n - j

                # check the two possible "beautiful" requirements
                ok = False
                if pref_ok and len2 >= len1:              # nums1 prefix of nums2
                    ok = True
                elif len2 <= len3 and equal_sub(i, j, len2):  # nums2 prefix of nums3
                    ok = True

                if ok:
                    ans += 1

        return ans