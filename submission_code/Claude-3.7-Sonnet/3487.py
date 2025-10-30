class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def is_subsequence(removed):
            # Check if pattern is still a subsequence after removing indices in removed
            j = 0  # Index for pattern
            
            for i in range(len(source)):
                if i in removed:
                    continue  # Skip removed characters
                
                if j < len(pattern) and source[i] == pattern[j]:
                    j += 1  # Match found, move to next pattern character
            
            return j == len(pattern)  # True if we matched the entire pattern
        
        operations = 0
        removed = set()
        
        for idx in targetIndices:
            removed.add(idx)  # Try removing this character
            if is_subsequence(removed):
                operations += 1  # Valid removal
            else:
                removed.remove(idx)  # Invalid removal, put it back
        
        return operations