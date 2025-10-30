from typing import List
from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # We transform the array into indicator values: 
        # f(x) = 1 if x % modulo == k, otherwise 0.
        # We need to count subarrays for which the sum of f(x) over the subarray, 
        # when taken modulo modulo, equals k.
        #
        # Let prefix[i] = sum of f(nums[0...i-1]). Then the subarray sum for subarray l..r is:
        #    subSum = prefix[r+1] - prefix[l]
        # And we need:
        #    (prefix[r+1] - prefix[l]) % modulo == k
        # This is equivalent to:
        #    prefix[r+1] % modulo == (prefix[l] + k) % modulo.
        #
        # We can compute prefix mods on the fly and count the number of valid pairs.
        
        # Dictionary to count occurrences of prefix modulo values.
        mod_count = defaultdict(int)
        # Starting with the "empty prefix": prefix sum = 0.
        mod_count[0] = 1
        
        prefix = 0
        result = 0
        
        # Process each element in nums.
        for num in nums:
            # Indicator function: 1 if condition satisfied, else 0.
            if num % modulo == k:
                prefix += 1
            # Compute current prefix mod.
            current_mod = prefix % modulo
            
            # For each current prefix, we need to find a previous prefix l such that:
            # current_mod == (prefix[l] + k) % modulo  or equivalently,
            # prefix[l] % modulo == (current_mod - k) mod modulo.
            needed = (current_mod - k) % modulo
            result += mod_count[needed]
            
            # Update the count of the current prefix mod.
            mod_count[current_mod] += 1
        
        return result