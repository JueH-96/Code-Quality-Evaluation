class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # We want all three final strings to be the same and non-empty.
        # Because we can only remove characters from the end, any final
        # common string must be a prefix of each of s1, s2, and s3.
        
        # Find the length of the longest common prefix among the three strings.
        min_len = min(len(s1), len(s2), len(s3))
        i = 0
        while i < min_len and s1[i] == s2[i] == s3[i]:
            i += 1
        
        # If no common prefix of length >= 1 is found, it's impossible to make
        # all three strings equal (and non-empty).
        if i == 0:
            return -1
        
        # Otherwise, the cost is the number of characters we need to remove
        # from the end of each string so that each becomes the common prefix.
        return (len(s1) - i) + (len(s2) - i) + (len(s3) - i)