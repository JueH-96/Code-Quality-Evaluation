class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        l = n - (numFriends - 1)
        max_sub = ""
        for i in range(n - l + 1):
            current = word[i:i + l]
            if current > max_sub:
                max_sub = current
        return max_sub