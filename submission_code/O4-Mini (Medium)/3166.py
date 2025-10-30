from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        # Count frequencies of each value
        from collections import Counter
        cnt = Counter(nums)
        freqs = list(cnt.values())
        n = len(nums)
        # We'll search for the minimal k in [1..n].
        # Groups k with same x = floor(n/k) form intervals [k, r].
        # For each such interval we test if there exists k in it
        # satisfying sum_v ceil(f_v/(x+1)) <= k <= sum_v floor(f_v/x).
        ans = n  # worst case each distinct, k = n
        k = 1
        while k <= n:
            x = n // k
            # the maximal k with floor(n/k) == x is r
            r = n // x
            # compute sumL = sum ceil(f/(x+1))
            xp1 = x + 1
            sumL = 0
            # sumL is minimal total groups needed
            for f in freqs:
                # ceil(f / (x+1))
                sumL += (f + x) // xp1
                # early break: if sumL already exceeds r, no k in [k..r] can match
                if sumL > r:
                    break
            if sumL <= r:
                # compute sumR = sum floor(f/x)
                sumR = 0
                for f in freqs:
                    sumR += f // x
                    # early break: if sumR already < k, no fit
                    if sumR < k and sumR + (len(freqs)-1) * (f // x) < k:
                        # this particular early break is tricky; we simplify and don't break
                        pass
                # Now check if there's some k' in [k..r] that lies in [sumL..sumR]
                lo = max(k, sumL)
                hi = min(r, sumR)
                if lo <= hi:
                    ans = lo
                    break
            # move to next interval
            k = r + 1
        return ans

# Example usage:
# sol = Solution()
# print(sol.minGroupsForValidAssignment([3,2,3,2,3]))  # Output: 2
# print(sol.minGroupsForValidAssignment([10,10,10,3,1,1]))  # Output: 4