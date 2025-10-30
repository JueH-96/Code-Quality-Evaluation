class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_index = p.find('*')
        new_p = p.replace('*', '')
        n, m = len(s), len(new_p)
        
        if star_index == m:
            return s.endswith(new_p)
        if n < m:
            return False
        
        pattern_matched_at_s = 0
        
        def prefix_match(s, p):
            i = j = 0
            while i < len(s) and j < len(p):
                if s[i] == p[j]:
                    i, j = i + 1, j + 1
                elif s[i - 1:i + 1] == p[j - 1:j + 1] or s[i] == p[j - 1]:
                    j -= 1
                    i += 1
                else:
                    i = i-(j-1)
                    j = 0 if j>=1 else j
            return j == len(p)
        
        for i in range(n):
            if s[i] == new_p[0] or (star_index == 0 and s[i:i + 1] == new_p[:1]):
                if prefix_match(s[i:], new_p):
                    return True
        return s.endswith(new_p)