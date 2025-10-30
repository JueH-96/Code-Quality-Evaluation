class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star = p.find('*')
        prefix = p[:star]
        suffix = p[star+1:]
        
        if not prefix and not suffix:
            return True
        if not prefix:
            return suffix in s
        if not suffix:
            return prefix in s
        
        pre_len = len(prefix)
        suf_len = len(suffix)
        
        max_i = len(s) - pre_len
        for i in range(max_i + 1):
            if s[i:i+pre_len] == prefix:
                if suffix in s[i+pre_len:]:
                    return True
        return False