class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        pre, suf = p.split('*')
        pre_len = len(pre)
        suf_len = len(suf)
        n = len(s)
        
        if pre_len + suf_len > n:
            return False
            
        if pre_len == 0 and suf_len == 0:
            return True
            
        for i in range(0, n - pre_len + 1):
            if s[i:i+pre_len] == pre:
                for j in range(i + pre_len, n - suf_len + 1):
                    if s[j:j+suf_len] == suf:
                        return True
                        
        return False