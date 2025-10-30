class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        
        # Sort the target indices in descending order
        targetIndices.sort(reverse=True)
        
        max_removals = 0
        
        # Try removing each character at the target indices
        for idx in targetIndices:
            # Create a copy of the source string
            new_source = source[:idx] + source[idx+1:]
            
            # Check if the pattern is still a subsequence of the new source
            i, j = 0, 0
            while i < len(new_source) and j < m:
                if new_source[i] == pattern[j]:
                    j += 1
                i += 1
            
            # If the pattern is still a subsequence, increment the max removals
            if j == m:
                max_removals += 1
        
        return max_removals