class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Count frequency of each character
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        
        # Find the maximum frequency
        max_freq = max(freq.values())
        
        # Find characters with maximum frequency and their last positions
        result = []
        for i in range(len(s) - 1, -1, -1):  # Traverse from right to left
            char = s[i]
            if freq[char] == max_freq:
                result.append(char)
                freq[char] = 0  # Mark as processed to avoid duplicates
        
        # Reverse to get the correct order
        return ''.join(reversed(result))