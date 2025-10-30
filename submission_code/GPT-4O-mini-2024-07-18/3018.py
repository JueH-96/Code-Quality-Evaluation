class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        n, m = len(str1), len(str2)
        if m > n:
            return False
        
        # Two pointers for str1 and str2
        j = 0
        
        for i in range(n):
            if j < m and str1[i] <= str2[j]:
                j += 1
            elif j < m and str1[i] > str2[j]:
                # Check if we can increment str1[i] to match str2[j]
                if (ord(str1[i]) - ord('a') + 1) % 26 == (ord(str2[j]) - ord('a')):
                    j += 1
        
        return j == m