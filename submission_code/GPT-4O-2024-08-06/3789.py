class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        # Total number of subarrays in an array of length n
        total_subarrays = n * (n + 1) // 2
        
        # Function to calculate the number of subarrays containing both a and b
        def count_subarrays_with_pair(a, b):
            if a > b:
                a, b = b, a
            # Subarrays containing both a and b must start at or before a and end at or after b
            return a * (n - b + 1)
        
        # Calculate the maximum number of subarrays possible after removing one conflicting pair
        max_non_conflicting_subarrays = 0
        
        for i in range(len(conflictingPairs)):
            # Remove the i-th conflicting pair
            remaining_pairs = conflictingPairs[:i] + conflictingPairs[i+1:]
            
            # Calculate the number of subarrays containing any of the remaining conflicting pairs
            subarrays_with_conflicts = 0
            for a, b in remaining_pairs:
                subarrays_with_conflicts += count_subarrays_with_pair(a, b)
            
            # Calculate the number of non-conflicting subarrays
            non_conflicting_subarrays = total_subarrays - subarrays_with_conflicts
            
            # Update the maximum number of non-conflicting subarrays
            max_non_conflicting_subarrays = max(max_non_conflicting_subarrays, non_conflicting_subarrays)
        
        return max_non_conflicting_subarrays