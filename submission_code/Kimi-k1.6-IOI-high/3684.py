class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        star_index = p.index('*')
        left = p[:star_index]
        right = p[star_index + 1:]
        min_length = len(left) + len(right)
        s_length = len(s)
        
        if min_length > s_length:
            return False
        
        for L in range(min_length, s_length + 1):
            for i in range(s_length - L + 1):
                substr = s[i:i+L]
                if substr.startswith(left) and substr.endswith(right):
                    return True
        return False