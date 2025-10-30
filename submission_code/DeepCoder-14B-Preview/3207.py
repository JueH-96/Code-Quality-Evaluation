class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        len1, len2, len3 = len(s1), len(s2), len(s3)
        min_len = min(len1, len2, len3)
        
        for t in range(min_len, 0, -1):
            substr1 = s1[:t]
            substr2 = s2[:t]
            substr3 = s3[:t]
            if substr1 == substr2 == substr3:
                return (len1 - t) + (len2 - t) + (len3 - t)
        
        return -1