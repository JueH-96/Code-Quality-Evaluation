class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # Convert array to 1s and 0s based on condition
        n = len(nums)
        
        # Use prefix sum approach
        prefix_mod_count = {0: 1}  # prefix sum 0 occurs once (empty prefix)
        current_sum = 0
        result = 0
        
        for num in nums:
            # Add 1 if num % modulo == k, else add 0
            if num % modulo == k:
                current_sum += 1
            
            # We want (current_sum - prev_sum) % modulo == k
            # This means current_sum % modulo == (prev_sum + k) % modulo
            # So we look for prev_sum where prev_sum % modulo == (current_sum - k) % modulo
            target = (current_sum - k) % modulo
            
            if target in prefix_mod_count:
                result += prefix_mod_count[target]
            
            # Add current prefix sum to count
            current_mod = current_sum % modulo
            prefix_mod_count[current_mod] = prefix_mod_count.get(current_mod, 0) + 1
        
        return result