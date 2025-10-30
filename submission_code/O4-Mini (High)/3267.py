class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        letters = set(s)
        # Try lengths from largest to smallest for an early exit
        for L in range(n, 0, -1):
            for c in letters:
                sub = c * L
                count = 0
                # Count overlapping occurrences of sub in s
                for i in range(n - L + 1):
                    if s[i:i+L] == sub:
                        count += 1
                        if count >= 3:
                            return L
        return -1