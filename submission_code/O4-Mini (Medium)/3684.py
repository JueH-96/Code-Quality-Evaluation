class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        n = len(s)
        star = p.index('*')
        before = p[:star]
        after = p[star+1:]
        lenB, lenA = len(before), len(after)
        # The matched substring must be non-empty:
        min_len = max(1, lenB + lenA)
        
        # Try every possible starting index i for the substring in s
        for i in range(0, n - lenB + 1):
            # Check if 'before' matches at position i
            if s[i:i+lenB] != before:
                continue
            # Try every possible end position j (exclusive)
            # Ensure substring length >= min_len
            for j in range(i + min_len, n + 1):
                # Check if 'after' matches right before j
                if s[j-lenA:j] == after:
                    return True
        return False