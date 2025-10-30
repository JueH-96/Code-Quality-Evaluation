class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        
        # Check all possible substrings of length k
        for i in range(n - k + 1):
            # Get the substring
            substring = s[i:i+k]
            
            # Check if all characters in substring are the same
            if len(set(substring)) == 1:
                char = substring[0]
                
                # Check character before (if exists)
                if i > 0 and s[i-1] == char:
                    continue
                
                # Check character after (if exists)
                if i + k < n and s[i+k] == char:
                    continue
                
                # All conditions satisfied
                return True
        
        return False