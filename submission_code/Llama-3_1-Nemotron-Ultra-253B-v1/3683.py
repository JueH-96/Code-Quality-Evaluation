class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        L = len(word) - numFriends + 1
        max_sub = ''
        for i in range(len(word) - L + 1):
            current = word[i:i+L]
            if current > max_sub:
                max_sub = current
        return max_sub