class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        # If there's only one friend, the only part is the whole word.
        if numFriends == 1:
            return word
        
        # Maximum allowed length of any segment is n - (numFriends - 1)
        # because the other (numFriends - 1) parts each need at least one character.
        max_len = n - numFriends + 1
        
        best = ""  # to track the best (lex largest) substring
        # For each start i, take the substring word[i : i+max_len] (or to end)
        # That will be the lex-largest among all substrings starting at i
        # with length up to max_len.
        for i in range(n):
            L = n - i
            if L > max_len:
                L = max_len
            if L <= 0:
                continue
            cand = word[i:i+L]
            if cand > best:
                best = cand
        return best