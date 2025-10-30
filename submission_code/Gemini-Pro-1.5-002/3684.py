class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_index = p.find('*')
        if star_index == -1:
            return s.find(p) != -1

        prefix = p[:star_index]
        suffix = p[star_index + 1:]

        if len(prefix) > len(s) or (len(suffix) > len(s) and len(suffix) != 0):
            return False
        
        if s.startswith(prefix) == False:
            return False

        if len(suffix) == 0:
            return True
        
        if s.endswith(suffix) == False:
            return False
        
        if len(prefix) + len(suffix) > len(s):
            return False
        
        return True