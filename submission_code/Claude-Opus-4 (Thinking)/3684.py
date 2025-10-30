class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split pattern by '*'
        star_idx = p.index('*')
        prefix = p[:star_idx]
        suffix = p[star_idx + 1:]
        
        prefix_len = len(prefix)
        suffix_len = len(suffix)
        n = len(s)
        
        # Try all possible substrings of s
        for i in range(n):
            for j in range(i + prefix_len + suffix_len, n + 1):
                # Check if s[i:j] matches the pattern
                # Check prefix
                if s[i:i + prefix_len] == prefix:
                    # Check suffix
                    if suffix_len == 0 or s[j - suffix_len:j] == suffix:
                        return True
        
        return False