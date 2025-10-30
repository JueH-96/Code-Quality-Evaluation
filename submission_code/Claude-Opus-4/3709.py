class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        
        # If k is greater than the length of string, no valid substring exists
        if k > n:
            return False
        
        # Check each possible substring of length k
        for i in range(n - k + 1):
            # Get the substring
            substring = s[i:i + k]
            
            # Check if all characters in substring are the same
            if len(set(substring)) == 1:
                char = substring[0]
                
                # Check character before (if exists)
                if i > 0 and s[i - 1] == char:
                    continue
                
                # Check character after (if exists)
                if i + k < n and s[i + k] == char:
                    continue
                
                # If we reach here, we found a valid substring
                return True
        
        return False