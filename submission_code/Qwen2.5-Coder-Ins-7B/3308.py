class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import Counter
        
        # Count the frequency of each character
        count = Counter(s)
        
        # Find the maximum frequency
        max_freq = max(count.values())
        
        # Collect characters that have the maximum frequency
        max_chars = [char for char, freq in count.items() if freq == max_freq]
        
        # Collect the last occurrence of each character that has the maximum frequency
        last_occurrences = []
        for char in reversed(s):
            if char in max_chars:
                last_occurrences.append(char)
                max_chars.remove(char)
                if not max_chars:
                    break
        
        # Return the string formed by the last occurrences in reverse order
        return ''.join(reversed(last_occurrences))