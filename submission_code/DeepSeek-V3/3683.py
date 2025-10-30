class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1:
            return word
        if numFriends >= len(word):
            return max(word)
        # Generate all possible splits into numFriends parts
        # Since the order of the parts doesn't matter, we can consider the first part
        # and find the lexicographically largest possible first part
        # The lexicographically largest string is the one that is as large as possible
        # in the first part, then the second, etc.
        # So, to maximize the first part, we can take as many characters as possible
        # from the start, but ensuring that the remaining parts can be split into
        # numFriends-1 parts.
        # The maximum possible length for the first part is len(word) - (numFriends - 1)
        max_len = len(word) - (numFriends - 1)
        # We need to find the largest possible substring of length up to max_len
        # that can be the first part of the split
        # To get the lexicographically largest string, we can take the first max_len characters
        # but if there are multiple options, we need to choose the one that is lex largest
        # So, we can iterate over all possible lengths from 1 to max_len and choose the largest
        # possible substring
        max_sub = ''
        for l in range(1, max_len + 1):
            current_sub = word[:l]
            if current_sub > max_sub:
                max_sub = current_sub
        return max_sub