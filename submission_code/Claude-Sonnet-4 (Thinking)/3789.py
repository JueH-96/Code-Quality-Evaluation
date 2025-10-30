class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        def count_valid_subarrays(pairs):
            total_subarrays = n * (n + 1) // 2
            
            if not pairs:
                return total_subarrays
            
            invalid_count = 0
            
            # Inclusion-exclusion principle
            for mask in range(1, 1 << len(pairs)):
                min_val = float('inf')
                max_val = 0
                
                # Find the intersection of forbidden regions
                for i in range(len(pairs)):
                    if mask & (1 << i):
                        a, b = pairs[i]
                        min_val = min(min_val, min(a, b))
                        max_val = max(max_val, max(a, b))
                
                # Size of intersection
                intersection_size = min_val * (n - max_val + 1)
                
                # Add or subtract based on inclusion-exclusion
                if bin(mask).count('1') % 2 == 1:
                    invalid_count += intersection_size
                else:
                    invalid_count -= intersection_size
            
            return total_subarrays - invalid_count
        
        max_count = 0
        
        # Try removing each conflicting pair
        for i in range(len(conflictingPairs)):
            remaining_pairs = conflictingPairs[:i] + conflictingPairs[i+1:]
            count = count_valid_subarrays(remaining_pairs)
            max_count = max(max_count, count)
        
        return max_count