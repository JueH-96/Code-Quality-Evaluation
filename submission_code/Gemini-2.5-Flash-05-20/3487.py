import collections
from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        L = len(targetIndices)

        # Helper function to check if 'pattern' is a subsequence of 'source'
        # after removing characters at indices specified by 'k_removals'.
        # According to observed behavior from examples, we try to remove the last k_removals
        # elements from targetIndices. This is the greedy choice that makes the check
        # function work for the binary search.
        def check(k_removals: int) -> bool:
            # If 0 removals are requested, pattern is always a subsequence (given by problem constraints).
            if k_removals == 0:
                return True

            # Create a set of indices that are conceptually removed.
            # We take the last k_removals elements from targetIndices.
            # Example: targetIndices = [0,1,2,3,4], k_removals = 2
            # Indices to remove are targetIndices[5-2:] which is targetIndices[3:] = [3,4]
            indices_to_remove = set(targetIndices[L - k_removals:])

            p_ptr = 0  # Pointer for current character in pattern
            s_ptr = 0  # Pointer for current character in source

            # Iterate through the source string
            while s_ptr < n and p_ptr < m:
                # If the current source character's index is among those to be removed, skip it.
                if s_ptr in indices_to_remove:
                    s_ptr += 1
                    continue
                
                # If the current source character matches the current pattern character, advance pattern pointer.
                if source[s_ptr] == pattern[p_ptr]:
                    p_ptr += 1
                
                # Always advance source pointer.
                s_ptr += 1
            
            # If all characters of pattern have been found, it's a subsequence.
            return p_ptr == m

        # Binary search for the maximum number of removable characters.
        low = 0
        high = L  # The maximum possible removals is L (all targetIndices).
        ans = 0   # Stores the maximum k for which check(k) is True.

        while low <= high:
            mid = low + (high - low) // 2 # Calculate mid-point to avoid overflow for very large low/high

            if check(mid):
                # If mid removals are possible, then mid is a candidate for the answer.
                # Try to find if even more removals are possible.
                ans = mid
                low = mid + 1
            else:
                # If mid removals are not possible, we need to try fewer removals.
                high = mid - 1
        
        return ans