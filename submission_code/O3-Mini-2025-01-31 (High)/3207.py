class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find the longest common prefix (lcp) among the three strings.
        # Since the only allowed operation is to delete characters from the right,
        # the final string must be a prefix of each string.
        min_length = min(len(s1), len(s2), len(s3))
        lcp_length = 0
        for i in range(min_length):
            if s1[i] == s2[i] and s2[i] == s3[i]:
                lcp_length += 1
            else:
                break
                
        # If there is no common prefix of length >= 1, we can't make them equal.
        if lcp_length == 0:
            return -1
        
        # The number of operations needed for one string is the difference between its current length
        # and the length of the final common prefix.
        # Total operations is the sum of required deletions from all three strings.
        return (len(s1) - lcp_length) + (len(s2) - lcp_length) + (len(s3) - lcp_length)

# For quick testing, you can use the following code:
if __name__ == "__main__":
    sol = Solution()
    # Example 1: Expected output is 2.
    print(sol.findMinimumOperations("abc", "abb", "ab"))
    # Example 2: Expected output is -1.
    print(sol.findMinimumOperations("dac", "bac", "cac"))