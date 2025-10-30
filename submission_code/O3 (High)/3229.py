from typing import List
import bisect

class Solution:
    # will be created once and re-used by every call
    _pal_list: List[int] = None

    @staticmethod
    def _build_pal_list() -> List[int]:
        """Generate every palindromic number 1 … 999_999_999."""
        if Solution._pal_list is not None:        # already built
            return Solution._pal_list

        limit = 10 ** 9                           # upper bound (exclusive)
        pals = []

        # create palindromes with 1 … 9 digits
        for length in range(1, 10):
            half_len = (length + 1) // 2          # number of digits we have to choose freely
            start = 10 ** (half_len - 1)          # first half cannot start with 0
            end = 10 ** half_len                  # one past the last

            for half in range(start, end):
                s = str(half)
                if length & 1:                    # odd number of digits
                    pal_s = s + s[-2::-1]         # mirror without the last digit
                else:                             # even number of digits
                    pal_s = s + s[::-1]           # mirror whole string
                val = int(pal_s)
                if val >= limit:                  # we must stay < 1e9
                    break
                pals.append(val)

        pals.sort()
        Solution._pal_list = pals
        return pals

    @staticmethod
    def _cost(sorted_nums: List[int], prefix: List[int], y: int) -> int:
        """
        Compute Σ|nums[i] – y| in O(log n) using binary search
        and a prefix sum on the sorted array.
        """
        n = len(sorted_nums)
        idx = bisect.bisect_left(sorted_nums, y)          # elements < y
        left  = y * idx - prefix[idx]                     # distance to the left part
        right = (prefix[n] - prefix[idx]) - y * (n - idx) # distance to the right part
        return left + right

    def minimumCost(self, nums: List[int]) -> int:
        pals = self._build_pal_list()

        # sort nums and build prefix sums
        arr = sorted(nums)
        n = len(arr)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + arr[i]

        # median interval [lowMed, highMed]
        lowMed  = arr[(n - 1) // 2]
        highMed = arr[n // 2]

        best = None

        # 1) is there a palindrome inside the median interval?
        pos = bisect.bisect_left(pals, lowMed)
        if pos < len(pals) and pals[pos] <= highMed:
            best = self._cost(arr, prefix, pals[pos])

        # 2) closest palindrome strictly below the interval
        if pos > 0:
            cand = pals[pos - 1]
            val = self._cost(arr, prefix, cand)
            best = val if best is None or val < best else best

        # 3) closest palindrome strictly above the interval
        pos_hi = bisect.bisect_right(pals, highMed)
        if pos_hi < len(pals):
            cand = pals[pos_hi]
            val = self._cost(arr, prefix, cand)
            best = val if best is None or val < best else best

        # At least one palindrome (1 … 9) always exists, so best is set
        return best