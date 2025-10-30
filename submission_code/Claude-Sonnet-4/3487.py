class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        target_set = set(targetIndices)
        n, m = len(source), len(pattern)
        
        from functools import lru_cache
        
        @lru_cache(None)
        def dp(i, j, removed):
            # i: current index in source
            # j: current index in pattern we need to match
            # removed: number of characters removed so far
            
            # If we've matched all pattern characters, we can remove all remaining targetIndices
            if j == m:
                remaining_targets = sum(1 for idx in targetIndices if idx >= i)
                return removed + remaining_targets
            
            # If we've reached end of source but haven't matched pattern, invalid
            if i == n:
                return -1
            
            max_removals = -1
            
            # Option 1: If current index is in targetIndices, try removing it
            if i in target_set:
                result = dp(i + 1, j, removed + 1)
                if result != -1:
                    max_removals = max(max_removals, result)
            
            # Option 2: Keep current character
            if source[i] == pattern[j]:
                # Character matches pattern, advance both pointers
                result = dp(i + 1, j + 1, removed)
                if result != -1:
                    max_removals = max(max_removals, result)
            else:
                # Character doesn't match pattern, just advance source pointer
                result = dp(i + 1, j, removed)
                if result != -1:
                    max_removals = max(max_removals, result)
            
            return max_removals
        
        result = dp(0, 0, 0)
        return result if result != -1 else 0