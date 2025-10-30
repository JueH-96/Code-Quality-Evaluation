from typing import List
from collections import Counter
import math

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        # Count frequencies of each number
        counts = Counter(nums)
        freqs = list(counts.values())

        # Find the minimum frequency
        # The minimum possible group size `k` cannot be greater than the minimum frequency.
        # If k > min_f, then a value with frequency min_f cannot be partitioned into groups of size k or k+1,
        # because the smallest possible sum using sizes k and k+1 (with at least one group) is k (if using one group of size k)
        # or k+1 (if using one group of size k+1). Both are greater than min_f.
        min_f = min(freqs)

        # We are looking for the minimum total number of groups.
        # Let the group sizes be k and k+1 for some k >= 1.
        # For a frequency f, we need to partition it into groups of size k or k+1.
        # Let g be the total number of groups for frequency f. Let g_k be the number of groups of size k
        # and g_{k+1} be the number of groups of size k+1.
        # f = g_k * k + g_{k+1} * (k+1)
        # g = g_k + g_{k+1}
        # This system has non-negative integer solutions for g_k and g_{k+1}
        # if and only if there exists an integer g such that f / (k+1) <= g <= f / k.
        # Such an integer g exists if and only if ceil(f / (k+1)) <= floor(f / k).
        # Using integer division, this condition is (f + k) // (k + 1) <= f // k.
        # If this condition holds for a frequency f, the minimum number of groups for this frequency is ceil(f / (k+1)) = (f + k) // (k + 1).

        # We iterate through possible values of k, starting from the largest possible (min_f)
        # down to the smallest possible (1).
        # The total number of groups for a fixed k is sum(ceil(f / (k+1)) for all frequencies f).
        # The function sum(ceil(f / (k+1))) is a sum of decreasing functions of k, so it is decreasing.
        # To minimize the total number of groups, we should find the largest valid k.
        for k in range(min_f, 0, -1):
            is_valid_k = True
            current_total_groups = 0

            # Check if this k is valid for all frequencies.
            for f in freqs:
                # The condition for k to be valid for frequency f is ceil(f / (k+1)) <= floor(f / k).
                # If this condition is false for any frequency, this k is not possible.
                
                min_groups_for_f = (f + k) // (k + 1) # Equivalent to ceil(f / (k+1))
                max_groups_for_f = f // k             # Equivalent to floor(f / k)

                if min_groups_for_f > max_groups_for_f:
                    is_valid_k = False
                    break
                
                # If k is valid for this frequency, the minimum number of groups needed
                # is min_groups_for_f. We sum these up.
                current_total_groups += min_groups_for_f

            # If k is valid for all frequencies, it is the largest valid k we encountered so far
            # (because we are iterating downwards). The total groups calculated for this k
            # is the minimum possible.
            if is_valid_k:
                return current_total_groups

        # The loop will always find a valid k. k=1 is always valid because
        # ceil(f/2) <= floor(f/1) is true for any f >= 1.
        # Since constraints guarantee 1 <= nums.length, min_f >= 1, so k=1 is included.
        # This line should theoretically not be reached.
        return -1