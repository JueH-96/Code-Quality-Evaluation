class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        if numFriends == 1:
            return word
        if numFriends == n:
            return word[0]
        
        ans = ""
        for i in range(1, n):
            if n - i >= numFriends - 1:
                if word[:i] > ans:
                    ans = word[:i]
        return ans