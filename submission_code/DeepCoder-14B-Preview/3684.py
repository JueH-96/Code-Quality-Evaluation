class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_pos = p.index('*')
        prefix = p[:star_pos]
        suffix = p[star_pos + 1:]
        len_prefix = len(prefix)
        len_suffix = len(suffix)
        min_length = len_prefix + len_suffix
        n = len(s)
        
        for i in range(n):
            for j in range(i, n):
                substr = s[i:j+1]
                if len(substr) < min_length:
                    continue
                if substr.startswith(prefix) and substr.endswith(suffix):
                    return True
        return False