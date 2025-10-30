class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i = 0  # pointer for str2
        
        for char in str1:
            if i < len(str2):
                # Check if current character matches or can be incremented to match
                target = str2[i]
                
                # Direct match
                if char == target:
                    i += 1
                # Check if incrementing char gives us target
                elif chr((ord(char) - ord('a') + 1) % 26 + ord('a')) == target:
                    i += 1
        
        # Return true if we've matched all characters in str2
        return i == len(str2)