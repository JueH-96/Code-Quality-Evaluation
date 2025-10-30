class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        n1 = len(s1)
        n2 = len(s2)
        n3 = len(s3)

        # The length of the longest common prefix (LCP) is limited by the shortest string.
        min_s_len = min(n1, n2, n3)

        lcp_len = 0
        # Calculate the length of the LCP of s1, s2, and s3.
        for i in range(min_s_len):
            if s1[i] == s2[i] and s2[i] == s3[i]:
                lcp_len += 1
            else:
                # Mismatch found, LCP ends at index i-1 (its length is the current lcp_len).
                break
        
        # If lcp_len is 0, it means the first characters of the strings are not all identical.
        # (Constraints state string lengths are at least 1).
        # If first characters differ, they can't be made equal to any non-empty string.
        # They also can't be made equal to an empty string, because an operation requires
        # string length to be at least 2. A string of length 1 cannot be shortened further.
        # Thus, if the LCP is effectively empty (lcp_len == 0), it's impossible.
        if lcp_len == 0:
            return -1
        else:
            # The target string will be this LCP.
            # The number of operations for each string is its original length minus lcp_len.
            # For example, to change s1="abc" to LCP="a" (lcp_len=1):
            # Operations for s1: len("abc") - 1 = 3 - 1 = 2.
            # ("abc" -> "ab" -> "a"). These operations are valid as per problem statement,
            # because lcp_len >= 1 ensures the target string is not empty.
            # The sum of operations for all three strings is the minimum total.
            total_ops = (n1 - lcp_len) + (n2 - lcp_len) + (n3 - lcp_len)
            return total_ops