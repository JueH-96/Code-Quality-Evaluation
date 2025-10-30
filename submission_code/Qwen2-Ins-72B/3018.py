class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        m, n = len(str1), len(str2)
        i, j = 0, 0
        
        while i < m and j < n:
            char1, char2 = str1[i], str2[j]
            
            if char1 == char2 or (ord(char1) - ord('a') + 1) % 26 == ord(char2) - ord('a'):
                j += 1
            
            i += 1
        
        return j == n