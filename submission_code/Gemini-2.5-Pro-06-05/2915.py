from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        
        # A map to store frequencies of (prefix_sum % modulo).
        # Initialize with {0: 1} to handle subarrays starting at index 0,
        # corresponding to the empty prefix which has a sum of 0.
        prefix_mod_counts = {0: 1}
        
        ans = 0
        
        # running_prefix_sum stores the count of numbers `x` seen so far
        # such that `x % modulo == k`.
        running_prefix_sum = 0
        
        for num in nums:
            # Update the prefix sum if the current number satisfies the condition.
            if num % modulo == k:
                running_prefix_sum += 1
            
            # For a subarray ending at the current element, its count of special numbers
            # is `running_prefix_sum - prev_prefix_sum`.
            # Let the current sum be P_r and a previous sum be P_l.
            # We need `(P_r - P_l) % modulo == k`.
            # This is equivalent to `P_l % modulo == (P_r - k) % modulo`.
            
            # Calculate the target remainder we need for P_l.
            # Python's % operator correctly handles negative results, giving a
            # value in [0, modulo-1] if modulo is positive.
            target_rem = (running_prefix_sum - k) % modulo
            
            # Add the number of previously seen prefix sums that had this target remainder.
            # Each one corresponds to a valid start of an interesting subarray.
            ans += prefix_mod_counts.get(target_rem, 0)
            
            # Update the map with the current prefix_sum's remainder, making it
            # available for subsequent elements.
            current_rem = running_prefix_sum % modulo
            prefix_mod_counts[current_rem] = prefix_mod_counts.get(current_rem, 0) + 1
            
        return ans