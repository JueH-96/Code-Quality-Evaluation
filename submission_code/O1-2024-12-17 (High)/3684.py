class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Find the position of '*'
        star_index = p.index('*')
        
        # Split the pattern p into left (before '*') and right (after '*')
        left = p[:star_index]
        right = p[star_index+1:]
        
        # Check every substring of s to see if it matches the pattern
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                sub = s[i:j]
                # The substring must start with left, end with right,
                # and be at least long enough to accommodate both left and right parts.
                if sub.startswith(left) and sub.endswith(right) and len(sub) >= len(left) + len(right):
                    return True
        return False