from typing import List

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        def count_valid_subarrays(n, pairs):
            # Create a set to track positions of conflicting elements
            conflict_set = set()
            for a, b in pairs:
                conflict_set.add(a)
                conflict_set.add(b)
            
            # Count valid subarrays
            total_subarrays = 0
            last_conflict = 0
            
            for i in range(1, n + 1):
                if i in conflict_set:
                    # If we hit a conflict, calculate the number of valid subarrays
                    total_subarrays += (i - last_conflict) * (i - last_conflict + 1) // 2
                    last_conflict = i
            
            # Add the remaining subarrays after the last conflict
            total_subarrays += (n - last_conflict) * (n - last_conflict + 1) // 2
            
            return total_subarrays
        
        max_subarrays = 0
        
        # Try removing each conflicting pair one by one
        for i in range(len(conflictingPairs)):
            # Create a new list of pairs excluding the i-th pair
            new_pairs = conflictingPairs[:i] + conflictingPairs[i+1:]
            # Count valid subarrays with the new list of pairs
            max_subarrays = max(max_subarrays, count_valid_subarrays(n, new_pairs))
        
        return max_subarrays