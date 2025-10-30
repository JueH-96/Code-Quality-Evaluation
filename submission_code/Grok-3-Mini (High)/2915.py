from typing import List
import collections

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        ans = 0
        prefix = 0
        freq = {0: 1}  # Frequency map for prefix sum modulo
        
        for num in nums:
            # Compute indicator: 1 if num % modulo == k, else 0
            indicator = 1 if (num % modulo == k) else 0
            # Update prefix sum modulo modulo
            prefix = (prefix + indicator) % modulo
            # Compute the target for which prefix sum should be equal
            target = (prefix - k) % modulo
            # Add the number of subarrays ending here with the required property
            ans += freq.get(target, 0)
            # Update the frequency map with the current prefix sum
            freq[prefix] = freq.get(prefix, 0) + 1
        
        return ans