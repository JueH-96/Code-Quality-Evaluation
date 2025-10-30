class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        
        # For each starting position
        for i in range(n):
            # Character frequency map for current substring
            freq = {}
            
            # Expand substring from position i
            for j in range(i, n):
                # Add current character to frequency map
                char = s[j]
                freq[char] = freq.get(char, 0) + 1
                
                # Check if any character appears at least k times
                if freq[char] >= k:
                    # All substrings from s[i:j] to s[i:n] are valid
                    count += n - j
                    break
        
        return count