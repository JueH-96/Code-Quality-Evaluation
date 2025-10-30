class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        L = n - numFriends + 1
        if L <= 0:
            return ""
        max_str = word[0:L]
        for i in range(1, n - L + 1):
            current = word[i:i+L]
            if current > max_str:
                max_str = current
        return max_str