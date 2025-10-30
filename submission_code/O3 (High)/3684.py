class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        """
        Return True if, after replacing the single '*' in pattern p by an
        arbitrary (possibly empty) sequence of characters, p can become a
        substring of s.
        """
        # split pattern into the fixed part before and after the '*'
        star_pos = p.index('*')
        prefix = p[:star_pos]
        suffix = p[star_pos + 1:]

        n = len(s)

        # Enumerate every non-empty substring of s: s[i : j+1]
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j + 1]
                # the substring must be at least long enough to hold prefix+suffix
                if len(sub) < len(prefix) + len(suffix):
                    continue
                # check prefix and suffix conditions
                if sub.startswith(prefix) and sub.endswith(suffix):
                    return True

        return False