from collections import defaultdict

class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Calculate the frequency of each character
        freq = defaultdict(int)
        for c in s:
            freq[c] += 1
        max_freq = max(freq.values()) if freq else 0
        
        # Find the last occurrence of each character
        last_occurrence = {}
        for i, c in enumerate(s):
            last_occurrence[c] = i
        
        # Collect characters with maximum frequency
        max_chars = set()
        for c in freq:
            if freq[c] == max_freq:
                max_chars.add(c)
        
        # Build the result string
        result = []
        for i, c in enumerate(s):
            if c in max_chars and i == last_occurrence[c]:
                result.append(c)
        
        return ''.join(result)