class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        # dp[i] = minimum number of balanced substrings that partition s[i:]
        dp = [10**9] * (n + 1)
        dp[n] = 0  # base: empty string requires 0 partitions

        # Process from the back of the string:
        for i in range(n - 1, -1, -1):
            freq = [0] * 26  # frequency counter for letters 'a' to 'z'
            for j in range(i, n):
                idx = ord(s[j]) - ord('a')
                freq[idx] += 1

                # Check if current substring s[i:j+1] is balanced:
                # A balanced string has every character (that appears) with equal frequency.
                required = None
                balanced = True
                for count in freq:
                    if count > 0:
                        if required is None:
                            required = count
                        elif count != required:
                            balanced = False
                            break

                # If balanced, update the dp state.
                if balanced:
                    dp[i] = min(dp[i], 1 + dp[j + 1])

        return dp[0]


# The following code is provided for testing the solution.
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    s1 = "fabccddg"
    print(sol.minimumSubstringsInPartition(s1))  # Expected output: 3

    # Example 2:
    s2 = "abababaccddb"
    print(sol.minimumSubstringsInPartition(s2))  # Expected output: 2