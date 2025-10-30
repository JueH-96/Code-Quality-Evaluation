class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Find the position of '*'
        star_pos = p.index('*')
        prefix = p[:star_pos]
        suffix = p[star_pos + 1:]
        
        # Try all possible starting positions in s
        for i in range(len(s) - len(prefix) + 1):
            # Check if prefix matches at position i
            if s[i:i + len(prefix)] == prefix:
                # Try all possible positions for suffix after the prefix
                # The suffix must start at or after position i + len(prefix)
                for j in range(i + len(prefix), len(s) - len(suffix) + 1):
                    if s[j:j + len(suffix)] == suffix:
                        return True
                
                # Special case: if suffix is empty, we found a match
                if len(suffix) == 0:
                    return True
        
        return False