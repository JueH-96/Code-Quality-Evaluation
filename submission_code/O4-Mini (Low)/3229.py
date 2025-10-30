from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Helper to build a palindrome of total length L from the integer prefix p.
        def make_palindrome(p: int, L: int) -> int:
            s = str(p)
            # Determine the expected length of the prefix
            if L % 2 == 0:
                # even length palindrome: prefix length = L//2
                if len(s) != L // 2:
                    return -1
                return int(s + s[::-1])
            else:
                # odd length palindrome: prefix length = (L+1)//2
                if len(s) != (L + 1) // 2:
                    return -1
                # omit the central digit when mirroring
                return int(s + s[-2::-1])

        n = len(nums)
        # Sort to find the median
        nums_sorted = sorted(nums)
        median = nums_sorted[n // 2]

        s_med = str(median)
        L = len(s_med)
        half_len = (L + 1) // 2

        # Extract the prefix from the median
        prefix = int(s_med[:half_len])

        candidates = set()

        # 1) Palindromes by tweaking the prefix by -1, 0, +1
        for delta in (-1, 0, 1):
            p = prefix + delta
            if p <= 0:
                continue
            y = make_palindrome(p, L)
            if 1 <= y < 10**9:
                candidates.add(y)

        # 2) The largest palindrome of length L-1: that's all 9s
        if L > 1:
            y = int("9" * (L - 1))
            if 1 <= y < 10**9:
                candidates.add(y)

        # 3) The smallest palindrome of length L+1: 10^L + 1
        y = 10**L + 1
        if y < 10**9:
            candidates.add(y)

        # Evaluate the cost for each candidate palindrome
        best = float('inf')
        for y in candidates:
            cost = 0
            # Sum of absolute deviations
            for v in nums:
                cost += abs(v - y)
                # Early break if cost already worse
                if cost >= best:
                    break
            if cost < best:
                best = cost

        return best