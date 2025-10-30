class Solution:
    def findValidPair(self, s: str) -> str:
        # Precompute a frequency dictionary of each digit in the string.
        freq = {}
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
        
        # Traverse s looking for adjacent pairs that are valid.
        for i in range(len(s) - 1):
            first, second = s[i], s[i+1]
            # They must be distinct.
            if first == second:
                continue
            # Check if the frequencies match the numeric value of the digit.
            if freq[first] == int(first) and freq[second] == int(second):
                return first + second
        return ""