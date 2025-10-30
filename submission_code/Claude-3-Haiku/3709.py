class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        
        for i in range(n - k + 1):
            substring = s[i:i+k]
            if len(set(substring)) == 1:
                if (i == 0 or s[i-1] != substring[0]) and (i + k == n or s[i+k] != substring[0]):
                    return True
        
        return False