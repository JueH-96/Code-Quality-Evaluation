class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        
        # The operation is to delete the rightmost character. This means any
        # resulting equal string must be a common prefix of the original strings.
        # To minimize deletions, we must find the longest common prefix (LCP).
        
        # The operation requires length >= 2, so strings cannot become empty.
        # This implies the target common prefix must be non-empty.
        # If the first characters don't match, no non-empty common prefix exists.

        # Find the length of the LCP. We only need to check up to the length
        # of the shortest string.
        lcp_len = 0
        min_len = min(len(s1), len(s2), len(s3))
        for i in range(min_len):
            if s1[i] == s2[i] and s2[i] == s3[i]:
                lcp_len += 1
            else:
                # The first mismatch marks the end of the common prefix.
                break
        
        # If lcp_len is 0, the first characters did not match, so it's impossible.
        if lcp_len == 0:
            return -1
            
        # The number of operations is the total number of characters to remove
        # to reduce each string to the LCP.
        ops1 = len(s1) - lcp_len
        ops2 = len(s2) - lcp_len
        ops3 = len(s3) - lcp_len
        
        return ops1 + ops2 + ops3