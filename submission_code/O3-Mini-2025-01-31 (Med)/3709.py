class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        for i in range(n - k + 1):
            # The substring from index i to i+k-1
            substr = s[i:i+k]
            # Check if all characters in the substring are the same
            if substr != substr[0] * k:
                continue
            
            # Check the left boundary condition if applicable
            if i > 0 and s[i - 1] == substr[0]:
                continue
            
            # Check the right boundary condition if applicable
            if i + k < n and s[i + k] == substr[0]:
                continue
            
            return True  # Found a valid special substring
        
        return False