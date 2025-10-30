class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        def count_valid_subarrays(pairs):
            valid_count = 0
            
            for start in range(1, n + 1):
                # Find the maximum valid ending position
                max_valid_end = n
                
                for a, b in pairs:
                    # Ensure a < b for consistency
                    if a > b:
                        a, b = b, a
                    
                    # If the subarray starts at or before a,
                    # it cannot end at or after b (to avoid containing both)
                    if start <= a:
                        max_valid_end = min(max_valid_end, b - 1)
                
                # Count valid subarrays starting at 'start'
                if max_valid_end >= start:
                    valid_count += max_valid_end - start + 1
            
            return valid_count
        
        max_valid = 0
        
        # Try removing each conflicting pair
        for i in range(len(conflictingPairs)):
            remaining_pairs = [conflictingPairs[j] for j in range(len(conflictingPairs)) if j != i]
            valid = count_valid_subarrays(remaining_pairs)
            max_valid = max(max_valid, valid)
        
        return max_valid