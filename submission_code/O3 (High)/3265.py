from typing import List
import math

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        """
        For every position j we want an earlier i (i < j) such that
            |nums[i] - nums[j]| = k   ->   nums[i] = nums[j] ± k
        
        Let  pref[t] be the sum of nums[0 .. t-1].
        Then the sum of sub-array nums[i .. j]  is
              pref[j+1] - pref[i]
        
        When the end index j is fixed, maximising the sub-array sum is
        equivalent to minimising pref[i].  Therefore, while scanning the
        array from left to right we keep, for every value v that has already
        appeared, the smallest prefix sum observed just before each
        occurrence of v.

        While at index j (with running prefix `pref_before_j`) we
        1. look-up the wanted values  nums[j] - k  and  nums[j] + k.
           If one of them has been seen, a good sub-array ending at j exists
           whose sum is
                 (pref_before_j + nums[j]) - best_pref_of_wanted_value
        2. update the dictionary for value  nums[j]  with the current
           prefix `pref_before_j`
        3. extend the running prefix by  nums[j]
        """
        # running prefix sum before processing the current element
        prefix_sum = 0

        # stores, for every value v already met, the minimal prefix_sum that
        # appeared *before* an occurrence of v
        best_prefix = dict()          # value v  ->  minimal prefix_sum

        found = False                 # did we meet at least one good sub-array?
        best = -math.inf              # maximal sum of a good sub-array

        for v in nums:
            t1 = v - k                # nums[i]  that gives  v - k
            t2 = v + k                # nums[i]  that gives  v + k

            if t1 in best_prefix:
                # sub-array starts right after the stored prefix of t1
                best = max(best, prefix_sum + v - best_prefix[t1])
                found = True
            if t2 in best_prefix:
                best = max(best, prefix_sum + v - best_prefix[t2])
                found = True

            # update dictionary for the current value
            # (prefix_sum is the sum up to, but not including, current v)
            if v not in best_prefix or prefix_sum < best_prefix[v]:
                best_prefix[v] = prefix_sum

            # move prefix_sum to include current element
            prefix_sum += v

        return best if found else 0