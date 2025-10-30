class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find the longest common prefix length among the three strings
        min_len = min(len(s1), len(s2), len(s3))
        lcp = 0
        for i in range(min_len):
            if s1[i] == s2[i] == s3[i]:
                lcp += 1
            else:
                break
        
        # If there is no non-empty common prefix, we cannot make them equal
        # because we cannot delete below length 1.
        if lcp == 0:
            return -1
        
        # Minimum operations is the total number of deletions needed
        # to reduce each string down to the common prefix of length lcp.
        return (len(s1) - lcp) + (len(s2) - lcp) + (len(s3) - lcp)