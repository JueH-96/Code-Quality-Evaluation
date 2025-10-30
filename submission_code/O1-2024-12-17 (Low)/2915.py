class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # c[i] = 1 if nums[i] % modulo == k, otherwise 0.
        # A subarray nums[l..r] is interesting if the sum(c[l..r]) % modulo == k.
        #
        # Define prefix_sum[i] as the sum of c up to index i-1 (i.e. c[0] + c[1] + ... + c[i-1]) modulo 'modulo'.
        # Then the subarray c[l..r] has sum = prefix_sum[r+1] - prefix_sum[l], taken modulo 'modulo'.
        # We want prefix_sum[r+1] - prefix_sum[l] ≡ k (mod modulo).
        # => prefix_sum[r+1] ≡ prefix_sum[l] + k (mod modulo).
        #
        # We can count how many prefix_sum[l] values satisfy that relation for each prefix_sum[r+1] we encounter.
        #
        # Steps to solve:
        # 1. Maintain a running prefix sum (modulo 'modulo') of c.
        # 2. Use a frequency map (dictionary) to keep track of how many times each prefix sum value has appeared.
        # 3. For the current prefix_sum value curr_ps, we want to find how many prefix sums prev_ps = curr_ps - k
        #    (mod modulo) have appeared so far.
        # 4. Add the count of such prefix sums to the result.
        # 5. Update the frequency map with the current prefix_sum value.
        #
        # This approach gives us the count of interesting subarrays in O(n) time.
        
        from collections import defaultdict
        
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1  # There's one way to have a prefix sum = 0 (before we start)
        
        prefix_sum = 0
        result = 0
        
        for num in nums:
            # Check if this element modifies the count
            if num % modulo == k:
                prefix_sum = (prefix_sum + 1) % modulo
            else:
                # Just keep the prefix_sum mod the same
                prefix_sum = prefix_sum % modulo
            
            # Compute the "needed" prefix sum that would make the subarray interesting.
            needed = (prefix_sum - k) % modulo
            
            # Add how many times 'needed' prefix sum has occurred
            result += prefix_counts[needed]
            
            # Update the frequency of the current prefix_sum
            prefix_counts[prefix_sum] += 1
        
        return result