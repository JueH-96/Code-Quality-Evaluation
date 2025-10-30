class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_pos = p.index('*')
        prefix = p[:star_pos]
        suffix = p[star_pos+1:]
        
        len_prefix = len(prefix)
        len_suffix = len(suffix)
        required_length = len_prefix + len_suffix
        
        for start in range(len(s)):
            for end in range(start, len(s)):
                substr = s[start:end+1]
                substr_len = end - start + 1
                if substr_len < required_length:
                    continue
                if substr.startswith(prefix) and substr.endswith(suffix):
                    return True
        return False