class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        
        # If pattern is longer than s, no valid substring exists
        if m > n:
            return -1
        
        # Check each possible starting position
        for i in range(n - m + 1):
            # Extract substring of length m starting at position i
            substring = s[i:i + m]
            
            # Count differences between substring and pattern
            differences = 0
            for j in range(m):
                if substring[j] != pattern[j]:
                    differences += 1
                    # Early termination if more than 1 difference
                    if differences > 1:
                        break
            
            # If at most 1 difference, we found a valid starting index
            if differences <= 1:
                return i
        
        # No valid starting index found
        return -1