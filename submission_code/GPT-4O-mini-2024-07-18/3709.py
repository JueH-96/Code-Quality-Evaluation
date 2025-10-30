class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        
        for i in range(n - k + 1):
            substring = s[i:i + k]
            if len(set(substring)) == 1:  # Check if all characters are the same
                # Check the character before and after the substring
                before_diff = (i == 0 or s[i - 1] != substring[0])
                after_diff = (i + k == n or s[i + k] != substring[0])
                
                if before_diff and after_diff:
                    return True
        
        return False