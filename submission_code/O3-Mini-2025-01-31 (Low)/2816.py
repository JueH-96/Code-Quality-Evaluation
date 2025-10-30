class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        # Convert string to list for easier manipulation.
        s_list = list(s)
        n = len(s_list)
        
        i, j = 0, n - 1
        while i < j:
            if s_list[i] != s_list[j]:
                # To use minimal operations, we only need to change one side.
                # To get lexicographically smallest palindrome, we choose the minimum of the two.
                smaller = min(s_list[i], s_list[j])
                s_list[i] = smaller
                s_list[j] = smaller
            i += 1
            j -= 1
        
        # Return the modified string.
        return "".join(s_list)
        
# For local testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    s1 = "egcfe"
    print(sol.makeSmallestPalindrome(s1))  # Expected "efcfe"
    # Example 2
    s2 = "abcd"
    print(sol.makeSmallestPalindrome(s2))  # Expected "abba"
    # Example 3
    s3 = "seven"
    print(sol.makeSmallestPalindrome(s3))  # Expected "neven"