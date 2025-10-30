class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Count total frequency of each character
        freq = [0] * 26
        base = ord('a')
        for ch in s:
            freq[ord(ch) - base] += 1
        
        # The number of rounds is the maximum frequency
        R = max(freq)
        
        # Track how many times we've seen each character so far
        seen = [0] * 26
        result = []
        
        # The string just before the last (R-th) round consists of
        # exactly the R-th occurrence of each character whose total
        # frequency is R, in the order they appear.
        for ch in s:
            idx = ord(ch) - base
            seen[idx] += 1
            # If this is the R-th time we've seen ch, and total freq is R,
            # then this character survives up to round R and is in the final string.
            if seen[idx] == freq[idx] == R:
                result.append(ch)
        
        return ''.join(result)