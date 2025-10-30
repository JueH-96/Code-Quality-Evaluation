class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        # Maximum allowed length for any piece is n - (numFriends - 1)
        max_len = n - (numFriends - 1)
        # Find the global maximum character in the word
        max_char = max(word)
        best = ""
        # For every position where word[i] == max_char,
        # consider the substring of length up to max_len
        for i, ch in enumerate(word):
            if ch == max_char:
                # take up to max_len characters from this suffix
                candidate = word[i : i + max_len]
                if candidate > best:
                    best = candidate
        return best