class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        L = n - (numFriends - 1)
        max_str = ""
        for i in range(n - L + 1):
            current = word[i:i+L]
            if current > max_str:
                max_str = current
        return max_str