class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        
        for i in range(n - k + 1):
            # Check if the substring of length k starting at i has all same characters
            if all(s[i] == s[j] for j in range(i, i + k)):
                # Check the character before the substring if it exists
                if i > 0 and s[i - 1] == s[i]:
                    continue
                # Check the character after the substring if it exists
                if i + k < n and s[i + k] == s[i]:
                    continue
                return True
        
        return False