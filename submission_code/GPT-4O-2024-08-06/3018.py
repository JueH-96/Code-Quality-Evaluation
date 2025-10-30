class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Pointers for str1 and str2
        i, j = 0, 0
        n, m = len(str1), len(str2)
        
        # Traverse both strings
        while i < n and j < m:
            # Check if current character of str1 matches str2 or can be incremented to match
            if str1[i] == str2[j] or (chr((ord(str1[i]) - ord('a') + 1) % 26 + ord('a')) == str2[j]):
                j += 1
            i += 1
        
        # If we have matched all characters of str2, return True
        return j == m