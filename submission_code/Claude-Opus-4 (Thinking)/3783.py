class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        from functools import lru_cache
        
        @lru_cache(None)
        def count_alternating(available_mask, last_is_odd):
            # Count alternating permutations with given available numbers
            # last_is_odd: None (first position), True, or False
            
            if available_mask == 0:
                return 1
            
            count = 0
            for i in range(n):
                if available_mask & (1 << i):
                    num = i + 1
                    num_is_odd = (num % 2 == 1)
                    
                    # Check if this number can be placed
                    if last_is_odd is None or num_is_odd != last_is_odd:
                        new_mask = available_mask ^ (1 << i)
                        count += count_alternating(new_mask, num_is_odd)
            
            return count
        
        def build_kth(available, k_remaining, last_is_odd):
            if not available:
                return []
            
            # Try each number in sorted order
            for num in sorted(available):
                num_is_odd = (num % 2 == 1)
                
                # Check if this number can be placed
                if last_is_odd is None or num_is_odd != last_is_odd:
                    # Calculate how many permutations start with this number
                    new_available = available - {num}
                    
                    # Convert to mask
                    mask = 0
                    for x in new_available:
                        mask |= (1 << (x - 1))
                    
                    count = count_alternating(mask, num_is_odd)
                    
                    if k_remaining < count:
                        # The k-th permutation starts with this number
                        return [num] + build_kth(new_available, k_remaining, num_is_odd)
                    else:
                        # Skip these permutations
                        k_remaining -= count
            
            return []  # Should not reach here
        
        # Check if k is valid
        full_mask = (1 << n) - 1
        total_count = count_alternating(full_mask, None)
        
        if k > total_count:
            return []
        
        # Build the k-th permutation (convert to 0-indexed)
        return build_kth(set(range(1, n + 1)), k - 1, None)