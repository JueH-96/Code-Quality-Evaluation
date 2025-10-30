class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import Counter
        
        # Count the frequency of each character in the string
        count = Counter(s)
        
        # Find the maximum frequency
        max_freq = max(count.values())
        
        # Create a set of characters that have the maximum frequency
        max_freq_chars = {char for char, freq in count.items() if freq == max_freq}
        
        # Traverse the string from the end to the beginning
        # and collect characters that are in the max_freq_chars set
        result = []
        seen = set()
        for char in reversed(s):
            if char in max_freq_chars and char not in seen:
                result.append(char)
                seen.add(char)
        
        # The result is the characters in reverse order
        return ''.join(reversed(result))