class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        K = numFriends
        n = len(word)
        if K == 1:
            return word
        max_len = n - K + 1
        max_sub = ''
        for i in range(n - max_len + 1):
            current_sub = word[i:i+max_len]
            if current_sub > max_sub:
                max_sub = current_sub
        return max_sub