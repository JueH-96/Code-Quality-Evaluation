class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        from collections import defaultdict
        
        # Prefix sum of counts of elements satisfying nums[i] % modulo == k
        prefix_count = 0
        # Dictionary to store the frequency of each prefix_count % modulo
        count_modulo = defaultdict(int)
        count_modulo[0] = 1  # Base case: there's one way to have a zero count before starting
        interesting_count = 0
        
        for num in nums:
            if num % modulo == k:
                prefix_count += 1
            
            # We want to find subarrays where the count of valid elements % modulo == k
            # This is equivalent to finding prefix_count % modulo == k
            mod_val = prefix_count % modulo
            
            # Add to the interesting count the number of times (mod_val - k) has appeared
            # since (prefix_count - previous_prefix_count) % modulo == k
            # which simplifies to prefix_count % modulo - previous_prefix_count % modulo == k
            # hence previous_prefix_count % modulo == prefix_count % modulo - k
            target_mod = (mod_val - k) % modulo
            interesting_count += count_modulo[target_mod]
            
            # Update the count of current mod_val in the dictionary
            count_modulo[mod_val] += 1
        
        return interesting_count