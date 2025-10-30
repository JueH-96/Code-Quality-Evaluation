class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Check if the two strings have the same characters
        if sorted(s1) != sorted(s2):
            return False
        
        # Check if the strings can be made equal using the given operation
        for i in range(len(s1) - 2):
            if s1[i] != s2[i] or s1[i+2] != s2[i+2]:
                for j in range(i+1, len(s1)):
                    if s1[i] == s2[j] and s1[i+2] == s2[j-2]:
                        s1 = s1[:i] + s1[j-2:j] + s1[i+1:j-2] + s1[i:i+2] + s1[j:]
                        break
                else:
                    return False
        
        return True