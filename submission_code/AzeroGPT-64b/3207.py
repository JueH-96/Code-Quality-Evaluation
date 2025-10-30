class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        res1 = self.longestCommonPrefix(s1, s2)
        res2 = self.longestCommonPrefix(res1, s3)
        if len(res2) == 0:
            return -1
        else:
            return len(s1) + len(s2) + len(s3) - 3*len(res2)
    
    def longestCommonPrefix(self, s1, s2):
        i = 0
        while i < min(len(s1), len(s2)) and s1[i] == s2[i]:
            i += 1
        return s1[:i]