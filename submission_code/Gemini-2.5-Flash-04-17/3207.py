class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        """
        Calculates the minimum number of operations to make three strings equal
        by deleting the rightmost character of a string if its length is at least 2.

        Args:
            s1: The first string.
            s2: The second string.
            s3: The third string.

        Returns:
            The minimum number of operations required, or -1 if it's impossible
            to make the strings equal.
        """
        # Find the minimum length among the three strings.
        # The longest possible common prefix cannot exceed this length.
        min_len = min(len(s1), len(s2), len(s3))
        
        # Find the length of the longest common prefix (LCP) among the three strings.
        lcp_len = 0
        # Iterate character by character up to the minimum length.
        for i in range(min_len):
            # If the characters at the current index i are the same in all three strings,
            # the common prefix extends by one character.
            if s1[i] == s2[i] and s1[i] == s3[i]:
                lcp_len += 1
            else:
                # If a mismatch is found, the common prefix ends just before this index.
                # We can stop comparing further.
                break
                
        # If the length of the LCP is 0, it means the very first characters
        # of the strings are not all equal. Since operations only remove
        # characters from the right, the first character can never be changed.
        # Therefore, if the first characters are not equal, it's impossible
        # to make the strings equal.
        if lcp_len == 0:
            return -1
        
        # If a common prefix of length lcp_len > 0 exists, this LCP is the
        # longest possible string that can be obtained from s1, s2, and s3
        # by deleting characters from the right. To minimize operations,
        # we should aim to make all strings equal to this LCP.
        # The number of operations required to transform a string s_i into
        # the LCP (which has length lcp_len) is simply the number of characters
        # that need to be removed from the right, which is len(s_i) - lcp_len.
        # The total minimum operations is the sum of operations for each string.
        
        # Note on constraint: The operation requires string length >= 2 before deletion.
        # If the target LCP length is lcp_len > 0, a string s_i needs len(s_i) - lcp_len
        # operations. The lengths during the reduction process are
        # len(s_i), len(s_i)-1, ..., lcp_len + 1. The last operation reduces
        # the length from lcp_len + 1 to lcp_len. Since lcp_len >= 1, lcp_len + 1 >= 2,
        # so the operation is always valid until the string reaches length lcp_len.
        
        operations = (len(s1) - lcp_len) + (len(s2) - lcp_len) + (len(s3) - lcp_len)
        
        return operations