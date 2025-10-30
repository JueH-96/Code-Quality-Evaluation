class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        N = len(word)
        K = numFriends
        ans = ""
        for S in range(N):
            a = 1 if S > 0 else 0
            if a + 1 <= K - 1:
                max_E = min(N - 1, S + (N - K))
                substr = word[S : max_E + 1]
                if substr > ans:
                    ans = substr
            else:  # a + 1 > K - 1
                if a <= K - 1 and K - 1 <= S:
                    substr = word[S : N]
                    if substr > ans:
                        ans = substr
        return ans