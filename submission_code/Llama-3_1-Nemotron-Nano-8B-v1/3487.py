from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        # Sort the target indices in reverse order
        sorted_indices = sorted(targetIndices, reverse=True)
        removed = set()
        count = 0
        m = len(pattern)
        n = len(source)
        
        for idx in sorted_indices:
            # Create a temporary set of removed indices including the current idx
            temp_removed = removed.copy()
            temp_removed.add(idx)
            
            # Check if the pattern is a subsequence of the source with temp_removed indices excluded
            p_idx = 0
            for s_idx in range(n):
                if s_idx in temp_removed:
                    continue
                if p_idx < m and source[s_idx] == pattern[p_idx]:
                    p_idx += 1
                    if p_idx == m:
                        break  # All characters in pattern are matched
            if p_idx == m:
                # If valid, add to removed and increment count
                removed.add(idx)
                count += 1
        
        return count