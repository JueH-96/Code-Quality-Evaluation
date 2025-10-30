class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        
        for i in range(n):
            char_count = {}
            for j in range(i, n):
                # Add character s[j] to the current substring
                char_count[s[j]] = char_count.get(s[j], 0) + 1
                
                # Check if the newly added character has reached the threshold
                if char_count[s[j]] >= k:
                    # All substrings from position i to any position >= j are valid
                    count += n - j
                    break
        
        return count