class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        # Convert string to list for mutability.
        s_list = list(s)
        n = len(s_list)
        i = 0
        j = n - 1
        
        # Use two pointers, one from start and one from the end.
        while i < j:
            if s_list[i] != s_list[j]:
                # Choose lexicographically smaller letter.
                min_char = min(s_list[i], s_list[j])
                s_list[i] = min_char
                s_list[j] = min_char
            i += 1
            j -= 1

        return "".join(s_list)
        
# Example usage:
# sol = Solution()
# print(sol.makeSmallestPalindrome("egcfe"))  # Expected Output: "efcfe"
# print(sol.makeSmallestPalindrome("abcd"))   # Expected Output: "abba"
# print(sol.makeSmallestPalindrome("seven"))  # Expected Output: "neven"