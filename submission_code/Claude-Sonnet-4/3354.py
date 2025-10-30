class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Convert string to list for easier manipulation
        result = list(s)
        
        # Keep track of frequency of each character
        freq = [0] * 26  # for 'a' to 'z'
        
        # First pass: count existing characters
        for char in s:
            if char != '?':
                freq[ord(char) - ord('a')] += 1
        
        # Second pass: replace '?' with optimal characters
        for i in range(len(result)):
            if result[i] == '?':
                # Find the character with minimum frequency
                # If tie, choose lexicographically smallest (which comes first in our iteration)
                min_freq = min(freq)
                for j in range(26):
                    if freq[j] == min_freq:
                        result[i] = chr(ord('a') + j)
                        freq[j] += 1
                        break
        
        return ''.join(result)