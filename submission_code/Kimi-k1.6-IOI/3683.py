class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        max_length = n - numFriends + 1
        max_sub = ""
        for i in range(n - max_length + 1):
            current_sub = word[i:i+max_length]
            if current_sub > max_sub:
                max_sub = current_sub
        return max_sub