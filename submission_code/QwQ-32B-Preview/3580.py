class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        M = len(pattern)
        N = len(s)
        
        # Ensure s is longer than pattern
        if M >= N:
            return -1
        
        for i in range(N - M + 1):
            # Count mismatches between s[i:i+M] and pattern
            mismatch_count = sum(1 for a, b in zip(s[i:i+M], pattern) if a != b)
            if mismatch_count <= 1:
                return i
        return -1