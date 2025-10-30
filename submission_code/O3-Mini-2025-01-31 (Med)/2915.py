from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # We use the prefix count idea.
        # Let P[i] be the count of indices j in [0, i] such that nums[j] % modulo == k.
        # Then the count for subarray [l, r] is P[r] - (P[l-1] if l > 0 else 0).
        # The subarray [l, r] is interesting if (P[r] - (P[l-1] if l>0 else 0)) % modulo == k.
        # Rearranging, for each r we want:
        #   P[r] % modulo - (P[l-1] % modulo) ≡ k (mod modulo)
        # i.e., (P[l-1] % modulo) must equal (P[r] - k) % modulo.
        #
        # We'll iterate over the array, computing the prefix count,
        # and use a dictionary to count frequency of prefix counts mod modulo encountered so far.
        
        freq = defaultdict(int)
        # Before any elements, prefix sum P[-1] = 0, so remainder = 0.
        freq[0] = 1
        
        total_interesting = 0
        prefix = 0  # This is P[i], cumulative count
        
        for num in nums:
            # Increase prefix count if num satisfies condition
            if num % modulo == k:
                prefix += 1
            
            # Compute current prefix mod value.
            current_mod = prefix % modulo
            # desired previous prefix mod value should be:
            desired = (current_mod - k) % modulo
            total_interesting += freq.get(desired, 0)
            
            # update frequency for this prefix remainder.
            freq[current_mod] += 1
        
        return total_interesting