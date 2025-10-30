class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Calculate the frequency of each character
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        
        # Find the maximum frequency
        max_freq = max(freq.values()) if freq else 0
        
        # Collect characters that are the max_freq-th occurrence
        counts = {}
        result = []
        for c in s:
            counts[c] = counts.get(c, 0) + 1
            if counts[c] == max_freq:
                result.append(c)
        
        return ''.join(result)