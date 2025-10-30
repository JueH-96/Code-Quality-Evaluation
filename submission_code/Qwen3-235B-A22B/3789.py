from typing import List

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        m = len(conflictingPairs)
        max_valid = 0

        # Iterate each pair to exclude
        for exclude_index in range(m):
            current_valid = 0
            # For each s from 1 to n, compute the valid subarrays count
            for s in range(1, n + 1):
                min_r = float('inf')
                # Check all pairs except the excluded one
                for i in range(m):
                    if i == exclude_index:
                        continue
                    l, r = conflictingPairs[i]
                    if l >= s:
                        if r < min_r:
                            min_r = r
                
                if min_r == float('inf'):
                    # No pairs cover s, all subarrays valid
                    e_max = n
                    current_valid += max(0, e_max - s + 1)
                else:
                    e_max = min_r - 1
                    if e_max < s:
                        current_valid += 0
                    else:
                        current_valid += (e_max - s + 1)
            
            if current_valid > max_valid:
                max_valid = current_valid
        
        return max_valid