class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        # Iterate over all possible starting positions of a substring of length k
        for i in range(len(s) - k + 1):
            # Check if the substring consists of one distinct character
            substring = s[i:i+k]
            # Instead of using set, we can compare the substring to the repeated character
            if substring != substring[0] * k:
                continue  # Not all characters are the same
            
            # Check the left neighbor if it exists
            if i > 0 and s[i-1] == substring[0]:
                continue
            
            # Check the right neighbor if it exists
            if i + k < len(s) and s[i+k] == substring[0]:
                continue
            
            # If all conditions are satisfied, we can return True immediately
            return True
        
        # If no valid substring found, return False
        return False