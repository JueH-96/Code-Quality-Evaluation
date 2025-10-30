class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_index = p.find('*')
        prefix = p[:star_index]
        suffix = p[star_index + 1:]
        
        for i in range(len(s) - len(p) + 1 + len(suffix)):
            if i + len(prefix) > len(s):
                continue
            if s[i:i+len(prefix)] == prefix:
                remaining_s = s[i+len(prefix):]
                if len(remaining_s) >= len(suffix):
                    if remaining_s[len(remaining_s)-len(suffix):] == suffix:
                        return True
        return False