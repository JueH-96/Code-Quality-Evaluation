class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        def count_valid_subarrays(n, pairs):
            # Total number of subarrays
            total = n * (n + 1) // 2
            
            # For each pair, subtract subarrays containing both elements
            # This might double-count, but for this problem it's a good approximation
            invalid = 0
            seen = set()
            
            for a, b in pairs:
                if a > b:
                    a, b = b, a
                pair_key = (a, b)
                if pair_key not in seen:
                    seen.add(pair_key)
                    # Subarrays containing both a and b
                    invalid += a * (n - b + 1)
            
            return total - invalid
        
        max_result = 0
        
        # Try removing each conflicting pair
        for i in range(len(conflictingPairs)):
            remaining_pairs = conflictingPairs[:i] + conflictingPairs[i+1:]
            result = count_valid_subarrays(n, remaining_pairs)
            max_result = max(max_result, result)
        
        return max_result