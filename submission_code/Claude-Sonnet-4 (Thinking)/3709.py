class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        for i in range(len(s) - k + 1):
            char = s[i]
            
            # Check if all characters in the substring are the same
            if all(s[j] == char for j in range(i, i + k)):
                # Check the character before the substring
                if i > 0 and s[i-1] == char:
                    continue
                
                # Check the character after the substring
                if i + k < len(s) and s[i+k] == char:
                    continue
                
                return True
        
        return False