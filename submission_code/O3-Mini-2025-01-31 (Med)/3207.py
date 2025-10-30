class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # The idea is that in each operation, we are only allowed to remove the last character,
        # so a string can only be reduced to one of its prefixes (with minimum length 1).
        # Thus, for the three strings to be equal, they must all share a common prefix of 
        # at least one character. To minimize operations, we should aim to choose the longest 
        # common prefix that appears in all strings.
        #
        # For a given prefix of length k in s_i (i=1,2,3), the cost to reduce s_i to that prefix is:
        # (len(s_i) - k). The total cost is:
        #    (len(s1) - k) + (len(s2) - k) + (len(s3) - k)
        #            = len(s1) + len(s2) + len(s3) - 3*k.
        #
        # Therefore, we need to compute the maximum k (where 1 <= k <= min(len(s1), len(s2), len(s3)))
        # such that the first k characters of s1, s2, and s3 all match.
        # If no such k exists, we return -1.
        
        min_length = min(len(s1), len(s2), len(s3))
        # Try from the maximum possible k down to 1
        for k in range(min_length, 0, -1):
            if s1[:k] == s2[:k] == s3[:k]:
                return (len(s1) - k) + (len(s2) - k) + (len(s3) - k)
        return -1

# Example usage:
# sol = Solution()
# print(sol.findMinimumOperations("abc", "abb", "ab"))   # Expected output: 2
# print(sol.findMinimumOperations("dac", "bac", "cac"))   # Expected output: -1