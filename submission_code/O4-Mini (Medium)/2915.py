from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # We convert nums into a 0/1 array A where
        # A[i] = 1 if nums[i] % modulo == k, else 0.
        # We want the number of subarrays whose sum mod `modulo` == k.
        # Let prefix_mod be the running sum of A modulo `modulo`.
        # A subarray (l..r) is interesting iff
        #   (prefix_mod[r+1] - prefix_mod[l]) % modulo == k
        #   => prefix_mod[l] == (prefix_mod[r+1] - k) % modulo
        # We keep a count of how many times each prefix_mod value has appeared
        # and for each new prefix_mod we add counts[(prefix_mod - k) % modulo].
        
        counts = defaultdict(int)
        # prefix sum = 0 seen once (empty prefix before any element)
        counts[0] = 1
        
        prefix_mod = 0
        ans = 0
        
        for x in nums:
            # Update the 0/1 flag
            if x % modulo == k:
                prefix_mod = (prefix_mod + 1) % modulo
            
            # Count how many prior prefixes satisfy the condition
            target = (prefix_mod - k) % modulo
            ans += counts[target]
            
            # Record the current prefix_mod for future subarrays
            counts[prefix_mod] += 1
        
        return ans