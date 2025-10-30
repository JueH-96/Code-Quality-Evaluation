from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        m = len(pattern)
        n = len(source)
        
        # Convert targetIndices to a set for O(1) lookups
        target_set = set(targetIndices)
        
        # Initialize previous and current rows for DP
        previous = [0] * (n + 1)
        for j in range(1, n + 1):
            previous[j] = previous[j - 1] + (1 if j - 1 in target_set else 0)
        
        for i in range(1, m + 1):
            current = [0] * (n + 1)
            current[0] = float('inf')  # Cannot match non-empty pattern with empty source
            for j in range(1, n + 1):
                if pattern[i - 1] == source[j - 1]:
                    # Option 1: Use this position for matching
                    option1 = previous[j - 1] + (1 if j - 1 in target_set else 0)
                    # Option 2: Do not use this position
                    option2 = current[j - 1]
                    current[j] = min(option1, option2)
                else:
                    # Cannot use this position for matching, carry over the value
                    current[j] = current[j - 1]
            previous = current
        
        # The minimal number of target positions used in the matching
        min_used = previous[n]
        
        # The maximum number of removals is total targetIndices minus the minimal used
        max_removals = len(targetIndices) - min_used
        return max_removals