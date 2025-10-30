class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Count frequency of each character
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
            
        # Find max frequency
        max_freq = max(freq.values())
        
        # Keep only characters with max frequency
        # and get their last positions
        last_pos = {}
        for i, c in enumerate(s):
            if freq[c] == max_freq:
                last_pos[c] = i
                
        # Sort characters by their last position
        result = []
        for c in last_pos:
            result.append((last_pos[c], c))
        result.sort()
        
        # Return string formed by these characters
        return ''.join(c for _, c in result)