from typing import List
import bisect

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # To minimize work, iterate over the smaller array
        if len(arr1) > len(arr2):
            arr1, arr2 = arr2, arr1

        # Convert arr2 to a sorted list of strings
        arr2_s = sorted(str(x) for x in arr2)
        ans = 0

        # Helper to compute common prefix length of two strings
        def common_prefix_len(a: str, b: str) -> int:
            l = min(len(a), len(b))
            i = 0
            while i < l and a[i] == b[i]:
                i += 1
            return i

        # For each number in arr1, find where it would fit in arr2_s lexicographically
        for x in arr1:
            s = str(x)
            # If the length of s is not greater than our current best, it can't improve the result
            if len(s) <= ans:
                continue

            idx = bisect.bisect_left(arr2_s, s)
            # Check the neighbor at idx (if any)
            if idx < len(arr2_s):
                cp = common_prefix_len(s, arr2_s[idx])
                if cp > ans:
                    ans = cp
            # Check the neighbor before idx (if any)
            if idx > 0:
                cp = common_prefix_len(s, arr2_s[idx - 1])
                if cp > ans:
                    ans = cp

        return ans