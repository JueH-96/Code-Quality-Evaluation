class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        
        # If k is greater than string length, return False
        if k > n:
            return False
        
        # Check each possible substring of length k
        for i in range(n - k + 1):
            # Get the first character of current substring
            curr_char = s[i]
            
            # Check if all characters in substring are same
            is_same = True
            for j in range(i, i + k):
                if s[j] != curr_char:
                    is_same = False
                    break
            
            if not is_same:
                continue
            
            # Check character before substring (if exists)
            if i > 0 and s[i-1] == curr_char:
                continue
                
            # Check character after substring (if exists)
            if i + k < n and s[i+k] == curr_char:
                continue
                
            # If we reach here, we found a valid substring
            return True
            
        return False