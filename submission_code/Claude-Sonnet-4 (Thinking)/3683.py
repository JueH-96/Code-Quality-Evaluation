class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        if numFriends == 1:
            return word
        
        max_string = ""
        
        for i in range(n):
            for j in range(i + 1, n + 1):
                # Check if word[i:j] can be part of some valid split
                if j <= n - numFriends + 1 or i >= numFriends - n + j - 1:
                    substring = word[i:j]
                    if substring > max_string:
                        max_string = substring
        
        return max_string