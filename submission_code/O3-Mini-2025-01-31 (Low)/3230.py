class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        # dp[i][c] will be the minimal number of changes needed to get a valid transformed
        # word[0...i] where word[i] is changed to letter c (as an integer from 0 to 25)
        dp = [[float('inf')] * 26 for _ in range(n)]
        
        # initialize for first character
        for c in range(26):
            cost = 0 if ord(word[0]) - ord('a') == c else 1
            dp[0][c] = cost
        
        # transition: For i from 1 to n-1
        for i in range(1, n):
            for cur in range(26):  # candidate letter for position i
                # calculate cost for current letter chosen for position i.
                cost = 0 if (ord(word[i]) - ord('a') == cur) else 1
                # try each previous letter from dp[i-1]
                for prev in range(26):
                    # check that prev letter and current letter are "not almost equal"
                    # They are almost equal if they are equal or adjacent in the alphabet.
                    if abs(prev - cur) <= 1:
                        continue
                    # update dp[i][cur] if we can use transition from prev -> cur.
                    dp[i][cur] = min(dp[i][cur], dp[i-1][prev] + cost)
        
        # answer is the minimal operations among all choices for the last character.
        return min(dp[n-1])
        
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.removeAlmostEqualCharacters("aaaaa"))   # Expected output: 2
    print(sol.removeAlmostEqualCharacters("abddez"))   # Expected output: 2
    print(sol.removeAlmostEqualCharacters("zyxyxyz"))  # Expected output: 3