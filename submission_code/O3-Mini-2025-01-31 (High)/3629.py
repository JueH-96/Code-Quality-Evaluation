class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        mod = 10**9 + 7
        # We define dp[i] as the length obtained by transforming the character
        # with index i ('a' is index 0, …, 'z' is index 25) after t transformations.
        # Base: With 0 transformations, every letter produces a string of length 1.
        dp = [1] * 26
        
        # Each transformation changes a letter according to:
        #  • If c is not 'z' (i.e. index i from 0 to 24), it becomes the next alphabet letter.
        #    So, f(c, step) = f(c+1, step-1)
        #  • If c is 'z' (i.e. index 25), it becomes "ab".
        #    So, f('z', step) = f('a', step-1) + f('b', step-1)
        #
        # Therefore, a transformation on our vector dp is:
        # dp_new[i] = dp[i+1] for i in 0..24, and dp_new[25] = dp[0] + dp[1].
        for _ in range(t):
            dp = dp[1:] + [(dp[0] + dp[1]) % mod]

        result = 0
        # The final answer is the sum over the values of dp at the positions
        # corresponding to each character in the input string s.
        for ch in s:
            result = (result + dp[ord(ch) - ord('a')]) % mod

        return result

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    # s = "abcyy", t = 2
    # Explanation:
    #  - t = 1: "abcyy" becomes "bcdzz"
    #  - t = 2: "bcdzz" becomes "cdeabab"
    #  => Final length = 7
    print(sol.lengthAfterTransformations("abcyy", 2))  # Expected output: 7

    # Example 2:
    # s = "azbk", t = 1
    # Explanation:
    #  - t = 1: "azbk" becomes "babcl"
    #  => Final length = 5
    print(sol.lengthAfterTransformations("azbk", 1))   # Expected output: 5