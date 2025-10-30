class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Frequency of every character
        freq = [0] * 26
        # Last position of every character
        last_pos = [-1] * 26
        
        for i, ch in enumerate(s):
            idx = ord(ch) - 97
            freq[idx] += 1
            last_pos[idx] = i
        
        # The number of complete rounds that will be performed
        max_freq = max(freq)
        
        # Characters that survive to the very last round
        survivors = []
        for i in range(26):
            if freq[i] == max_freq:
                # store (last position, character)
                survivors.append((last_pos[i], chr(i + 97)))
        
        # The order in the string just before the last operation
        # is the order of their final remaining occurrences
        survivors.sort()                     # sort by last position
        return ''.join(ch for _, ch in survivors)