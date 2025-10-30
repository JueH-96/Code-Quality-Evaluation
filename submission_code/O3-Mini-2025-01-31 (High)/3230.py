class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        # Represent letters as integers 0...25
        # Define a high value larger than maximum possible cost.
        INF = float('inf')
        
        # Initialize dp table: dp[i][c] is the minimum operations to get a valid string up to
        # index i, ending with letter c (where c is 0 for 'a', 1 for 'b', ...).
        dp = [[INF] * 26 for _ in range(n)]
        
        # For index 0, we can choose any letter: cost is 0 if it is the same as word[0], else 1.
        orig0 = ord(word[0]) - ord('a')
        for c in range(26):
            dp[0][c] = 0 if c == orig0 else 1
        
        # Process subsequent characters.
        for i in range(1, n):
            orig = ord(word[i]) - ord('a')
            for new_c in range(26):
                # Cost for choosing letter new_c at position i: 0 if not changed else 1.
                cost = 0 if new_c == orig else 1
                # Try all previous letters that can connect to new_c validly.
                for prev_c in range(26):
                    # The adjacent pair (prev_c, new_c) must not be "almost-equal". 
                    # They are almost equal if they are the same or differ by 1.
                    if abs(new_c - prev_c) > 1:
                        dp[i][new_c] = min(dp[i][new_c], dp[i-1][prev_c] + cost)
        
        # The answer is the minimum cost among the letters possible at the last position.
        return min(dp[n-1])
        
# Sample test cases to verify the solution
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    word1 = "aaaaa"
    print(sol.removeAlmostEqualCharacters(word1))  # Expected output: 2

    # Example 2
    word2 = "abddez"
    print(sol.removeAlmostEqualCharacters(word2))  # Expected output: 2

    # Example 3
    word3 = "zyxyxyz"
    print(sol.removeAlmostEqualCharacters(word3))  # Expected output: 3