from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        
        def check(k: int) -> bool:
            """
            Checks if pattern remains a subsequence of source after removing
            the last k elements from targetIndices.
            
            Args:
                k: The number of elements to remove from the end of targetIndices.
                
            Returns:
                True if pattern is still a subsequence, False otherwise.
            """
            n = len(source)
            m = len(pattern)
            target_len = len(targetIndices)
            
            # Create a set of indices to be removed. These are the last k elements
            # from the sorted targetIndices list.
            # targetIndices[-k:] gets the last k elements.
            # If k is 0, the slice is empty. If k > target_len, it correctly gets
            # all elements up to the end.
            removed_indices_set = set(targetIndices[target_len - k:])
            
            source_ptr = 0
            pattern_ptr = 0
            
            # Iterate through the original source string
            for source_ptr in range(n):
                # If the current index in source is one of the removed indices, skip this character
                if source_ptr in removed_indices_set:
                    continue
                    
                # If the current character in source matches the current character
                # we are looking for in the pattern
                if pattern_ptr < m and source[source_ptr] == pattern[pattern_ptr]:
                    pattern_ptr += 1 # Move to the next character in pattern
                
                # If we have found all characters of the pattern, we can stop early
                if pattern_ptr == m:
                    break
                    
            # If pattern_ptr reached the length of the pattern, we found the pattern
            # as a subsequence using the remaining characters.
            return pattern_ptr == m

        # Binary search for the maximum number of removals (k) that can be performed.
        # The number of removals k can range from 0 to len(targetIndices).
        target_len = len(targetIndices)
        low = 0
        high = target_len
        ans = 0 # Initialize the answer with the minimum possible removals (0)

        # The binary search finds the largest integer 'mid' in the range [low, high]
        # for which the condition 'check(mid)' is True.
        while low <= high:
            mid = (low + high) // 2 # The number of removals we are currently testing
            
            # Check if removing the last 'mid' elements from targetIndices
            # still allows pattern to be a subsequence of source.
            if check(mid):
                # If it is possible to remove 'mid' elements, then 'mid' is a
                # potential answer. We try to remove even more elements, so
                # we store 'mid' as a possible answer and search in the
                # upper half [mid + 1, high].
                ans = mid
                low = mid + 1
            else:
                # If it is not possible to remove 'mid' elements, then we have
                # removed too many. The answer must be less than 'mid', so we
                # search in the lower half [low, mid - 1].
                high = mid - 1

        # The variable 'ans' holds the maximum number of removals (the largest k)
        # for which check(k) returned True.
        return ans