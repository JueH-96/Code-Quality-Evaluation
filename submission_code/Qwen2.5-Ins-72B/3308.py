class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import Counter
        
        # Count the frequency of each character
        char_count = Counter(s)
        
        # Find the maximum frequency
        max_freq = max(char_count.values())
        
        # Filter characters that have the maximum frequency
        max_freq_chars = {char for char, count in char_count.items() if count == max_freq}
        
        # Initialize the result string
        result = []
        
        # Iterate over the string in reverse to keep the last occurrence of max frequency characters
        for char in reversed(s):
            if char in max_freq_chars:
                result.append(char)
                max_freq_chars.remove(char)
        
        # Return the result in the correct order
        return ''.join(reversed(result))