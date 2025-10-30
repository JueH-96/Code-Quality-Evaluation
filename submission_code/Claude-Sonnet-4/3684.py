class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Find the position of '*'
        star_pos = p.index('*')
        
        # Split pattern into prefix and suffix
        prefix = p[:star_pos]
        suffix = p[star_pos + 1:]
        
        # Check all possible substrings of s
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                substring = s[i:j]
                
                # Check if substring matches the pattern
                if len(substring) >= len(prefix) + len(suffix):
                    if substring.startswith(prefix) and substring.endswith(suffix):
                        # Make sure prefix and suffix don't overlap
                        if len(prefix) + len(suffix) <= len(substring):
                            return True
        
        return False