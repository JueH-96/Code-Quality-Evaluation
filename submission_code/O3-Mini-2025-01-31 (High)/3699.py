from math import gcd
import bisect
from collections import defaultdict
from typing import List

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        # We'll “decompose” a valid quadruple (p,q,r,s) into two pairs:
        #  • A "left‐pair" (p,q) chosen from indices with gap: q - p >= 2.
        #  • A "right‐pair" (r,s) chosen from indices with gap: s - r >= 2.
        #
        # The equation nums[p]*nums[r] == nums[q]*nums[s] is equivalent (dividing
        # both sides by nums[q]*nums[r], which are nonzero) to:
        #       nums[p] / nums[q] == nums[s] / nums[r].
        #
        # So if we compute the reduced fraction of the left‐pair as
        #    f_left = (nums[p] // g, nums[q] // g)   where g = gcd(nums[p], nums[q])
        # and the reduced “fraction” on the right‐pair (note the order: numerator from s)
        #    f_right = (nums[s] // g, nums[r] // g)   where g = gcd(nums[s], nums[r])
        # then the special condition holds if f_left == f_right.
        #
        # In addition we must ensure the overall order p < q < r < s.
        # We already have p and q from the left‐pair (with q - p >=2) and
        # r and s from the right‐pair (with s - r >=2).
        # Finally the overall quadruple order forces that the left pair’s second index q must come
        # at least 2 before the start r of the right‐pair, i.e. we need r >= q + 2.
        #
        # We can pre‐compute and group by fraction value:
        #   left_map[f] will be the list of indices “q” (the second index in (p,q)) 
        #     from all valid left‐pairs for which
        #         f = (nums[p]//g, nums[q]//g)
        #   right_map[f] will be the list of indices “r” (the first index in (r,s)) 
        #     from all valid right‐pairs for which
        #         f = (nums[s]//g, nums[r]//g)
        #
        # Then for each fraction f that appears in both maps, every pair of a left‐pair (with second index q)
        # and a right‐pair (with first index r), such that r >= q + 2, yields a valid quadruple.
        
        left_map = defaultdict(list)
        # Build left‐pairs (p,q) with p+2 ≤ q < n.
        for p in range(n):
            # q must be at least p+2 so that there is at least one element in between.
            for q in range(p + 2, n):
                # Compute the reduced fraction for (nums[p], nums[q])
                g = gcd(nums[p], nums[q])
                key = (nums[p] // g, nums[q] // g)
                # We store q (the second index) because later we need to enforce that
                # the first index r of the right‐pair satisfies r >= q+2.
                left_map[key].append(q)
                
        right_map = defaultdict(list)
        # Build right‐pairs (r,s) with r+2 ≤ s < n.
        for r in range(n):
            for s in range(r + 2, n):
                g = gcd(nums[s], nums[r])
                # Note: for the right pair we use (nums[s], nums[r]) so that
                # the fraction is nums[s] / nums[r].
                key = (nums[s] // g, nums[r] // g)
                # Store r (the first index in the right‐pair)
                right_map[key].append(r)
                
        ans = 0
        # For each fraction f that occurs in both groups, count the combinations
        # that satisfy r >= q + 2.
        for key, left_list in left_map.items():
            if key not in right_map:
                continue
            r_list = right_map[key]
            left_list.sort()
            r_list.sort()
            # For each left‐pair with second index q, count how many right‐pairs have r >= q + 2.
            for q in left_list:
                # find the first index in r_list with value >= q+2
                idx = bisect.bisect_left(r_list, q + 2)
                ans += (len(r_list) - idx)
        return ans