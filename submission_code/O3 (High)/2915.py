from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        """
        Let pref be the prefix-sum of “qualified” positions, where a position is
        qualified if nums[i] % modulo == k.  For every index r we need the number
        of l ≤ r such that (pref[r] − pref[l-1]) % modulo == k.
        
        Re-arranging gives     pref[l-1] % modulo == (pref[r] − k) % modulo.
        While scanning the array we store how many times every prefix value
        (mod `modulo`) has occurred so far, and sum the counts of the needed
        value.
        
        Complexity: O(n) time, O(min(n, modulo)) extra space.
        """
        freq = defaultdict(int)   # frequency of each prefix value modulo `modulo`
        freq[0] = 1               # empty prefix
        prefix = 0                # current prefix modulo `modulo`
        ans = 0

        for x in nums:
            # update prefix if current element is a 'hit'
            if x % modulo == k:
                prefix = (prefix + 1) % modulo

            # we need previous prefixes equal to (prefix - k) mod modulo
            need = (prefix - k) % modulo
            ans += freq.get(need, 0)

            freq[prefix] += 1

        return ans