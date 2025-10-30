class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        """
        We want to remove characters from 'source' at indices in 'targetIndices' while
        still keeping 'pattern' as a subsequence in the resulting 'source'. Return the
        maximum number of such removals.

        Approach:
        1) Use binary search on the number of removals, k, from 0..len(targetIndices).
        2) For a mid value = k, mark the first mid indices in targetIndices as removed.
        3) Check if 'pattern' is still a subsequence of 'source' when these positions
           are skipped (we do NOT shift indices in source).
        4) If 'pattern' can still be formed, move the search boundary up; otherwise, move it down.
        5) The final answer is the largest k for which 'pattern' remains a subsequence.
        """

        import bisect
        
        def can_preserve_subsequence(k):
            # Mark the first k indices from targetIndices as removed
            removed_set = set(targetIndices[:k])
            p_i = 0  # pointer for pattern

            for i, ch in enumerate(source):
                # if not removed, try to match the pattern
                if i not in removed_set:
                    if ch == pattern[p_i]:
                        p_i += 1
                        if p_i == len(pattern):
                            return True
            return p_i == len(pattern)

        # Binary search boundaries
        lo, hi = 0, len(targetIndices)
        ans = 0

        while lo <= hi:
            mid = (lo + hi) // 2

            if can_preserve_subsequence(mid):
                ans = mid  # mid is feasible, try for a higher value
                lo = mid + 1
            else:
                hi = mid - 1

        return ans