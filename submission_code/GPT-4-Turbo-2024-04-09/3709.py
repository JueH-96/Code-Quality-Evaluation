class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        if k > n:
            return False
        
        i = 0
        while i <= n - k:
            # Check if the substring from i to i+k is valid
            substring = s[i:i+k]
            if substring == substring[0] * k:  # Check if all characters are the same
                # Check the character before the substring if it exists
                if i > 0 and s[i-1] == substring[0]:
                    i += 1
                    continue
                # Check the character after the substring if it exists
                if i + k < n and s[i+k] == substring[0]:
                    i += 1
                    continue
                return True
            i += 1
        
        return False