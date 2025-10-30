class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1:
            return word
        
        max_len = len(word) - numFriends + 1
        return max(word[i:j] for i in range(len(word)) 
                   for j in range(i + 1, min(i + max_len + 1, len(word) + 1)))