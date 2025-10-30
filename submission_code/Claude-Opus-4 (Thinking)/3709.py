class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        
        # Check all possible substrings of length k
        for i in range(n - k + 1):
            substring = s[i:i+k]
            
            # Check if all characters in substring are the same
            if len(set(substring)) == 1:
                char = substring[0]
                
                # Check boundary conditions
                # Before: either no character before or it's different
                before_ok = (i == 0) or (s[i-1] != char)
                # After: either no character after or it's different
                after_ok = (i + k == n) or (s[i+k] != char)
                
                if before_ok and after_ok:
                    return True
        
        return False