class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        # dp[i] = maximum segments used to reach boundary i
        dp = [-1] * (n+1)
        dp[0] = 0  # starting from index 0 we haven't taken any segment
        
        best_candidate = ""
        
        # For each boundary i that is reachable (0 <= i < n),
        # try to choose the next segment.
        for i in range(n):
            r = dp[i]
            # skip if not reachable or already used all segments
            if r < 0 or r >= numFriends:
                continue
            # Next segment (r+1) must end at j with:
            #    i+1 <= j <= n - (numFriends - (r+1))
            jmax = n - (numFriends - (r+1))
            if jmax > n:
                jmax = n
            if i+1 > jmax:
                continue

            # For fixed starting index i, the lexicographically maximum segment is
            # the one that extends as far as possible.
            candidate = word[i:jmax]
            if candidate > best_candidate:
                best_candidate = candidate

            new_val = r + 1
            # update dp for boundaries j = i+1 up to jmax
            for j in range(i+1, jmax+1):
                if new_val > dp[j]:
                    dp[j] = new_val
        
        return best_candidate

# A few tests:
if __name__ == '__main__':
    sol = Solution()
    # Example 1
    print(sol.answerString("dbca", 2))  # Expected output: "dbc"
    # Example 2
    print(sol.answerString("gggg", 4))  # Expected output: "g"
    
    # Additional tests:
    print(sol.answerString("aba", 2))   # Expected output: "ba"
    print(sol.answerString("abcde", 3))  # You can verify this with your own reasoning.