class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        # n is the length of the word.
        n = len(word)
        if n == 0:
            return 0  # Although constraints state at least 1 character.
        
        # Initialize a large number for DP comparisons.
        INF = float('inf')
        
        # dp[i][c] will be the minimum cost (number of operations) needed to transform
        # the substring word[0..i] when we turn word[i] into the letter corresponding to c (0-indexed for 'a')
        # There are 26 lowercase English letters.
        dp = [[INF] * 26 for _ in range(n)]
        
        # Base case: convert first character to any letter c.
        # If the chosen letter is the same as word[0], cost is 0; otherwise cost is 1.
        for c in range(26):
            dp[0][c] = 0 if (ord(word[0]) - ord('a') == c) else 1
        
        # Helper function: check if two letters (represented by their numeric values 0-25)
        # are "almost equal". They are almost equal if they are the same or adjacent in the alphabet.
        def almost_equal(c1: int, c2: int) -> bool:
            return abs(c1 - c2) <= 1
        
        # Fill dp for positions 1 to n-1.
        for i in range(1, n):
            for c in range(26):
                # For the current position i, if we decide to change to character corresponding to c,
                # then check all possibilities for the previous character.
                # We need to ensure that the previous character (pc) and the current character c
                # are not almost equal.
                for pc in range(26):
                    if not almost_equal(pc, c):
                        # Cost for position i: if the letter c is the same as word[i], cost is 0; else 1.
                        cost = 0 if (ord(word[i]) - ord('a') == c) else 1
                        dp[i][c] = min(dp[i][c], dp[i-1][pc] + cost)
        
        # The answer will be the minimum value achieved at the last position.
        result = min(dp[-1])
        return result


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Test Example 1:
    word = "aaaaa"
    print(sol.removeAlmostEqualCharacters(word))  # Expected Output: 2

    # Test Example 2:
    word = "abddez"
    print(sol.removeAlmostEqualCharacters(word))  # Expected Output: 2

    # Test Example 3:
    word = "zyxyxyz"
    print(sol.removeAlmostEqualCharacters(word))  # Expected Output: 3