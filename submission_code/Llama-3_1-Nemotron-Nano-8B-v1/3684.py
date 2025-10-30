class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_idx = p.find('*')
        prefix = p[:star_idx]
        suffix = p[star_idx+1:]
        min_len = len(prefix) + len(suffix)
        
        if min_len == 0:
            return len(s) >= 1
        
        for i in range(len(s) - min_len + 1):
            for j in range(i + min_len - 1, len(s)):
                substr = s[i:j+1]
                if substr.startswith(prefix) and substr.endswith(suffix):
                    return True
        return False