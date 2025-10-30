from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        """
        For every number build the (small) set of all values that can be obtained by
        performing at most one swap of two digits in that number (the original number
        itself is included because we may apply zero operations).
        A pair (i , j) is almost-equal if
              nums[j] is in reachable[i]   or   nums[i] is in reachable[j].
        With n ≤ 100 and at most 7 digits per number the whole procedure is tiny.
        """
        # Pre-compute all numbers reachable from each element with one swap
        reach = []
        for num in nums:
            s = list(str(num))
            cur = {num}                         # zero swap (original value)
            m = len(s)
            if m >= 2:
                for i in range(m):
                    for j in range(i + 1, m):
                        t = s[:]                # copy of the digits
                        t[i], t[j] = t[j], t[i] # swap two positions
                        cur.add(int(''.join(t)))# leading zeros vanish here
            reach.append(cur)

        # Count qualifying pairs
        res, n = 0, len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if nums[j] in reach[i] or nums[i] in reach[j]:
                    res += 1
        return res