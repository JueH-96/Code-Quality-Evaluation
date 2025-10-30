class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        """
        Scan every substring of length k. 
        The substring is valid if:
          1. all k characters are identical
          2. the character just before it (if any) is different
          3. the character just after it (if any) is different
        """
        n = len(s)
        if k > n:                          # quick guard (though k ≤ n by constraints)
            return False
        
        for i in range(n - k + 1):
            segment = s[i:i + k]
            
            # condition 1: all the same
            if len(set(segment)) != 1:
                continue                   # not uniform, skip
            
            ch = segment[0]
            
            # condition 2: different from previous character (if any)
            if i > 0 and s[i - 1] == ch:
                continue
            
            # condition 3: different from next character (if any)
            if i + k < n and s[i + k] == ch:
                continue
            
            # all conditions satisfied
            return True
        
        return False