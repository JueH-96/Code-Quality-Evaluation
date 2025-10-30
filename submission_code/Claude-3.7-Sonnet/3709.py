class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        
        for i in range(n - k + 1):
            # Extract the substring
            substring = s[i:i+k]
            
            # Check if the substring consists of only one distinct character
            if len(set(substring)) == 1:
                char = substring[0]
                
                # Check if the character immediately before the substring (if it exists) is different
                before_condition = i == 0 or s[i-1] != char
                
                # Check if the character immediately after the substring (if it exists) is different
                after_condition = i + k == n or s[i+k] != char
                
                if before_condition and after_condition:
                    return True
        
        return False