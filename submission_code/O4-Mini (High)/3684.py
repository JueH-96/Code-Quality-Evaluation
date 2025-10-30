class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        n = len(s)
        # split pattern around the single '*'
        star_idx = p.index('*')
        prefix = p[:star_idx]
        suffix = p[star_idx+1:]
        a, b = len(prefix), len(suffix)
        
        # try every starting position for the prefix
        for i in range(0, n - a + 1):
            if s[i:i+a] != prefix:
                continue
            # j is the position where the suffix would start
            start_j = i + a
            # ensure we don't run past the end of s
            end_j = n - b if b > 0 else n
            # try every valid j
            for j in range(start_j, end_j + 1):
                if s[j:j+b] == suffix:
                    return True
        return False