class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        t = list(s)
        # First, count the frequency of each character in the fixed positions
        freq = [0] * 26
        for i in range(n):
            if t[i] != '?':
                freq[ord(t[i]) - ord('a')] += 1
        # Now, for each '?', choose the character with the smallest frequency
        # and update the frequency
        for i in range(n):
            if t[i] == '?':
                # Find the character with the smallest frequency
                min_freq = float('inf')
                min_char = 'a'
                for c in range(26):
                    if freq[c] < min_freq:
                        min_freq = freq[c]
                        min_char = chr(c + ord('a'))
                # Assign the character with the smallest frequency
                t[i] = min_char
                # Update the frequency
                freq[ord(min_char) - ord('a')] += 1
        return ''.join(t)