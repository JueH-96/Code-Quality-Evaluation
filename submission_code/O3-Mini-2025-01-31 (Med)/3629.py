MOD = 10**9 + 7

class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        # dp[i] will represent the contribution (length) of a letter with index i
        # ('a' is 0, ..., 'z' is 25) after a certain number of transformations.
        # Initially, each letter contributes 1 (0 transformations).
        dp = [1] * 26
        
        # Process t transformations.
        for _ in range(t):
            new_dp = [0] * 26
            # For letters 'a' to 'y' (i = 0 to 24), each becomes the next letter.
            # So, contribution of letter i becomes the contribution of letter i+1 from previous step.
            for i in range(25):
                new_dp[i] = dp[i+1] % MOD
            # For letter 'z' (i = 25), it is replaced by "ab".
            # Its contribution becomes the sum of contributions of 'a' and 'b'.
            new_dp[25] = (dp[0] + dp[1]) % MOD
            dp = new_dp
        
        # Sum up the contributions for each starting character in s.
        result = 0
        for ch in s:
            index = ord(ch) - ord('a')
            result = (result + dp[index]) % MOD
        
        return result

# Example usage (for testing):
if __name__ == '__main__':
    sol = Solution()
    # Test case 1:
    # s = "abcyy", t = 2
    # After transformations: "cdeabab", length = 7
    print(sol.lengthAfterTransformations("abcyy", 2))  # Output: 7
    
    # Test case 2:
    # s = "azbk", t = 1
    # After transformations: "babcl", length = 5
    print(sol.lengthAfterTransformations("azbk", 1))   # Output: 5