class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # Count of prefix sums modulo 'modulo'
        prefix_count = {0: 1}  # Empty prefix has count 0
        
        current_count = 0  # Running count of elements where nums[i] % modulo == k
        result = 0
        
        for num in nums:
            # Update the running count
            if num % modulo == k:
                current_count += 1
            
            # Current prefix sum modulo 'modulo'
            current_mod = current_count % modulo
            
            # We need to find previous positions where:
            # (current_mod - prev_mod) % modulo == k
            # Which means: prev_mod == (current_mod - k) % modulo
            target = (current_mod - k) % modulo
            
            # Add count of valid subarrays ending at current position
            if target in prefix_count:
                result += prefix_count[target]
            
            # Update the prefix count map
            prefix_count[current_mod] = prefix_count.get(current_mod, 0) + 1
        
        return result