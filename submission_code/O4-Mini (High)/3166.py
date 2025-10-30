from typing import List
from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        # Count the frequency of each number
        freq = Counter(nums)
        # Build a frequency-of-frequency map: how many values appear exactly f times
        freq_of_freq = Counter(freq.values())
        # Convert to list of (frequency, count_of_values_with_that_frequency)
        freq_counts = list(freq_of_freq.items())
        
        n = len(nums)
        best = None
        
        # We iterate over k in segments where a = floor(n/k) is constant.
        # Standard trick: for L from 1 to n, let a = n//L, r = n//a,
        # then for all k in [L..r], floor(n/k) == a.
        L = 1
        while L <= n:
            a = n // L
            R = n // a  # all k in [L..R] have floor(n/k) == a
            b = a + 1
            
            # Compute S_min = sum_v ceil(f_v / b)
            # and    S_max = sum_v floor(f_v / a)
            S_min = 0
            S_max = 0
            for f, cnt_vals in freq_counts:
                # minimum groups needed for frequency f if group-size max is b
                S_min += cnt_vals * ((f + b - 1) // b)
                # maximum groups possible for frequency f if group-size min is a
                S_max += cnt_vals * (f // a)
            
            # We need k in [L..R] ∩ [S_min..S_max]
            low_k = max(L, S_min)
            high_k = min(R, S_max)
            if low_k <= high_k:
                # the smallest feasible k in this segment is low_k
                if best is None or low_k < best:
                    best = low_k
            
            L = R + 1
        
        # As a fallback (should not happen), each index in its own group => k = n
        return best if best is not None else n