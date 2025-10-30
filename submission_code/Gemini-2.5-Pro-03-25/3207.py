import math # Import is not strictly needed as min() is built-in, but good practice if other math functions were used.

class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        """
        Calculates the minimum number of operations to make s1, s2, and s3 equal.
        An operation consists of deleting the rightmost character of a string
        if its length is at least 2.

        Args:
            s1: The first string.
            s2: The second string.
            s3: The third string.

        Returns:
            The minimum number of operations required to make the three strings equal,
            or -1 if it's impossible.
            
        Example 1:
        Input: s1 = "abc", s2 = "abb", s3 = "ab"
        Output: 2
        Explanation: Delete 'c' from s1 -> "ab". Delete 'b' from s2 -> "ab". s3 is already "ab". Total 2 operations.

        Example 2:
        Input: s1 = "dac", s2 = "bac", s3 = "cac"
        Output: -1
        Explanation: The first characters 'd', 'b', 'c' are different. They can never be made equal by deleting from the right.
        """
        len1 = len(s1)
        len2 = len(s2)
        len3 = len(s3)
        
        # Determine the minimum length among the three strings. This sets the upper
        # bound for the length of the common prefix we need to check. Any common prefix
        # cannot be longer than the shortest string.
        min_len = min(len1, len2, len3)
        
        # Find the length of the longest common prefix (LCP) among the three strings.
        # Initialize the length of the LCP to 0.
        lcp_length = 0
        # Iterate through the indices from 0 up to (but not including) min_len.
        for i in range(min_len):
            # Check if the characters at the current index `i` are the same across all three strings.
            # The chained comparison `s1[i] == s2[i] == s3[i]` efficiently checks this in Python.
            if s1[i] == s2[i] == s3[i]:
                # If the characters match, it means the common prefix extends at least to this index.
                # Increment the length of the common prefix found so far.
                lcp_length += 1
            else:
                # If the characters at index `i` do not match, the common prefix ends
                # just before this index (i.e., its length is `lcp_length` calculated so far). 
                # We stop searching further as any subsequent characters cannot be part of the common prefix.
                break
                
        # After the loop, lcp_length holds the length of the longest common prefix.
        
        # If the length of the LCP is 0, it signifies that the strings do not share
        # even the first character (given the problem constraints state lengths >= 1). 
        # In this scenario, it's impossible to make them equal by deleting characters from the right,
        # because the differing first characters will always remain as the leftmost characters.
        if lcp_length == 0:
            return -1
        else:
            # If a common prefix exists (lcp_length > 0), the only way to make the strings
            # equal is to reduce each of them to a common prefix. To minimize the
            # number of operations (which are deletions), we must target the *longest* common prefix,
            # as this requires the fewest deletions overall.
            # The total number of operations required is the sum of deletions needed for each string
            # to transform it into the LCP.
            # For s1, deletions needed = len1 - lcp_length
            # For s2, deletions needed = len2 - lcp_length
            # For s3, deletions needed = len3 - lcp_length
            # Total operations = sum of these deletions.
            operations = (len1 - lcp_length) + (len2 - lcp_length) + (len3 - lcp_length)
            
            # Return the calculated total minimum number of operations.
            return operations