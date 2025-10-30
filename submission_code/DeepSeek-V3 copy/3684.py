class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_index = p.find('*')
        prefix = p[:star_index]
        suffix = p[star_index+1:]
        
        for i in range(len(s) - len(prefix) - len(suffix) + 1):
            if s[i:i+len(prefix)] == prefix:
                remaining = s[i+len(prefix):]
                if len(remaining) >= len(suffix):
                    if remaining[-len(suffix):] == suffix:
                        return True
        return False