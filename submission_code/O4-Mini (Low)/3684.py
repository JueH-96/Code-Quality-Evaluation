class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split pattern at the single '*'
        star_idx = p.index('*')
        pre = p[:star_idx]
        suf = p[star_idx+1:]
        n = len(s)
        len_pre = len(pre)
        len_suf = len(suf)
        
        # Try every possible starting index for the prefix match
        for i in range(n - len_pre + 1):
            # Check if prefix matches at position i
            if s[i:i+len_pre] != pre:
                continue
            
            # Minimum end position j so that
            # substring = s[i:j] has at least pre + suf,
            # and is non-empty: j - i >= 1
            j_min = i + len_pre + len_suf
            # ensure non-empty substring
            j_min = max(j_min, i + 1)
            # Maximum j is n
            for j in range(j_min, n+1):
                # Check if suffix matches ending at j
                # i.e. s[j-len_suf:j] == suf
                if len_suf == 0 or s[j-len_suf:j] == suf:
                    return True
        
        return False