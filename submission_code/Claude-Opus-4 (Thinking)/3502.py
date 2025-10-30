class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        
        # Try each starting position
        for i in range(n):
            freq = {}
            
            # Expand substring starting at position i
            for j in range(i, n):
                char = s[j]
                freq[char] = freq.get(char, 0) + 1
                
                # If current character reaches k occurrences
                if freq[char] >= k:
                    # All substrings from s[i:j+1] to s[i:n] are valid
                    count += n - j
                    break
        
        return count