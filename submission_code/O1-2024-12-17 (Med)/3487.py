class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        """
        We want to remove the largest subset of indices from targetIndices while ensuring that
        'pattern' still remains a subsequence of 'source'.

        Approach (Binary Search):
          1. Use binary search on k = number of removed indices (0 .. len(targetIndices)).
          2. For a given k (midpoint in the binary search), we simulate removing the first k
             indices in targetIndices (since targetIndices is sorted), and check if 'pattern'
             is still a subsequence of the remaining string.
             - Set removed[ index ] = True for each index in targetIndices[:k].
             - Then scan through 'source', skipping removed indices, checking if we can match
               all characters of 'pattern' in order.
          3. If the pattern is still a subsequence, we can try to increase k (move the lower
             bound up). Otherwise, we decrease k (move the upper bound down).
          4. The highest k for which the pattern remains a subsequence is our answer.

        Time complexity: O(n log n), where n = len(source). This is feasible for n up to 3000.
        """

        # Helper function to check if 'pattern' is still
        # a subsequence when certain indices are removed.
        def can_still_form_pattern(k: int) -> bool:
            removed = [False] * len(source)
            # Mark the first k sorted target indices as removed
            for i in range(k):
                removed[targetIndices[i]] = True
            
            p_idx = 0  # pattern pointer
            for i, ch in enumerate(source):
                if removed[i]:
                    continue
                if ch == pattern[p_idx]:
                    p_idx += 1
                    if p_idx == len(pattern):
                        return True
            return p_idx == len(pattern)
        
        low, high = 0, len(targetIndices)
        answer = 0
        
        while low <= high:
            mid = (low + high) // 2
            if can_still_form_pattern(mid):
                answer = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return answer