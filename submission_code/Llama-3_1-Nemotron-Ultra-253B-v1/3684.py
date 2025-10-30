class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_pos = p.index('*')
        prefix = p[:star_pos]
        suffix = p[star_pos+1:]
        
        len_prefix = len(prefix)
        len_suffix = len(suffix)
        len_s = len(s)
        
        for i in range(len_s - len_prefix + 1):
            if s[i:i+len_prefix] == prefix:
                min_j = i + len_prefix
                max_j = len_s - len_suffix
                for j in range(min_j, max_j + 1):
                    if s[j:j+len_suffix] == suffix:
                        substring_length = j + len_suffix - i
                        if substring_length >= 1:
                            return True
        return False