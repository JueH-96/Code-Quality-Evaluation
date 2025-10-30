class Solution:
    def getSmallestString(self, s: str) -> str:
        """
        Check every possible single swap between adjacent digits that
        share the same parity (both even or both odd).  Keep the
        lexicographically smallest string encountered, including the
        original (no-swap) string.
        """
        best = s                      # start with the original string
        n = len(s)
        
        for i in range(n - 1):
            # Are s[i] and s[i+1] of the same parity?
            if (int(s[i]) % 2) == (int(s[i + 1]) % 2):
                # create the string resulting from swapping positions i and i+1
                chars = list(s)
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                cand = ''.join(chars)
                if cand < best:
                    best = cand        # keep the smaller one
        
        return best