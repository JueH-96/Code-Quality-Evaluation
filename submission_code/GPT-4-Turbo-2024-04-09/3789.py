class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        from collections import defaultdict
        
        def count_valid_subarrays(n, conflicts):
            """ Count valid subarrays given a set of conflicts. """
            # Create a mapping from each number to the numbers it conflicts with
            conflict_map = defaultdict(set)
            for a, b in conflicts:
                conflict_map[a].add(b)
                conflict_map[b].add(a)
            
            # Total subarrays calculation
            total_subarrays = n * (n + 1) // 2
            
            # To avoid subarrays with conflicting pairs
            invalid_subarrays = 0
            for num in range(1, n + 1):
                if num in conflict_map:
                    for conflict in conflict_map[num]:
                        if num < conflict:  # Ensure each pair is only considered once
                            # Calculate the number of invalid subarrays containing both num and conflict
                            # Positions of num: 1 to n
                            # Positions of conflict: 1 to n
                            # Valid positions for num and conflict together:
                            # num: 1 to n, conflict: num+1 to n (if conflict > num)
                            if conflict <= n:
                                # Number of positions for num is from 1 to conflict-1
                                # Number of positions for conflict is from conflict to n
                                num_start = 1
                                num_end = conflict - 1
                                conflict_start = conflict
                                conflict_end = n
                                
                                # Count subarrays starting from num_start to num_end that include num and conflict
                                num_count = num_end - num_start + 1
                                conflict_count = conflict_end - conflict_start + 1
                                
                                if num_count > 0 and conflict_count > 0:
                                    invalid_subarrays += num_count * conflict_count
            
            return total_subarrays - invalid_subarrays
        
        max_valid_subarrays = 0
        
        # Try removing each pair and calculate the maximum number of valid subarrays
        for i in range(len(conflictingPairs)):
            remaining_conflicts = conflictingPairs[:i] + conflictingPairs[i+1:]
            valid_subarrays = count_valid_subarrays(n, remaining_conflicts)
            max_valid_subarrays = max(max_valid_subarrays, valid_subarrays)
        
        return max_valid_subarrays