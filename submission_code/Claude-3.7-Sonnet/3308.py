class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Count the frequency of each character
        char_counts = {}
        for char in s:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Find the maximum frequency
        max_freq = max(char_counts.values())
        
        # We'll track how many times we've seen each character
        seen_counts = {}
        
        # Build the result by collecting the max_freq-th occurrence 
        # of each character with maximum frequency
        result = []
        
        for char in s:
            seen_counts[char] = seen_counts.get(char, 0) + 1
            # If this is the max_freq-th occurrence of a character with max frequency
            if seen_counts[char] == max_freq and char_counts[char] == max_freq:
                result.append(char)
        
        return ''.join(result)