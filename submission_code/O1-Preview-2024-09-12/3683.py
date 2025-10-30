class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        N = len(word)
        max_length = N - (numFriends - 1)
        max_substring = ''
        for L in range(1, max_length + 1):
            for i in range(N - L + 1):
                substring = word[i:i+L]
                if substring > max_substring:
                    max_substring = substring
        return max_substring