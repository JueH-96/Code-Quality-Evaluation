class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        
        # For each starting position
        for i in range(n):
            freq = {}
            # Expand the substring from position i
            for j in range(i, n):
                # Add current character to frequency map
                char = s[j]
                freq[char] = freq.get(char, 0) + 1
                
                # Check if any character has frequency >= k
                if freq[char] >= k:
                    # All substrings from i to j, j+1, j+2, ... n-1 are valid
                    count += n - j
                    break
        
        return count