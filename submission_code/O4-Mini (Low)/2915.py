from collections import defaultdict
from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # We transform the array into indicator values: 1 if nums[i] % modulo == k, else 0.
        # Then we want the number of subarrays whose sum % modulo == k.
        # Let prefix sum S[j] = sum of indicators up to (but excluding) index j.
        # For a subarray [l..r], the sum of indicators is S[r+1] - S[l].
        # We need (S[r+1] - S[l]) % modulo == k  <=>  S[r+1] % modulo == (S[l] + k) % modulo.
        # We sweep j = r+1 from 0 to n, keep a map of counts of previous S[i] % modulo,
        # and for each current S[j] % modulo we add count[(S[j] - k) % modulo].
        
        count = defaultdict(int)
        # initial prefix sum = 0
        prefix_mod = 0
        count[prefix_mod] = 1
        
        ans = 0
        for x in nums:
            # indicator: does x % modulo == k?
            if x % modulo == k:
                prefix_mod = (prefix_mod + 1) % modulo
            # else prefix_mod stays the same
            
            # we want previous_prefix_mod % modulo == (prefix_mod - k) % modulo
            target = (prefix_mod - k) % modulo
            ans += count[target]
            
            # record this prefix_mod for future subarrays
            count[prefix_mod] += 1
        
        return ans