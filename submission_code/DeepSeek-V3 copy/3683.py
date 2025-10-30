class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1:
            return word
        if numFriends >= len(word):
            return max(word)
        # Generate all possible splits
        # Since the order of splits doesn't matter, we can consider the first split
        # and then take the lexicographically largest part
        # To get the lexicographically largest string, we need to find the largest possible substring
        # that can be formed by splitting the word into numFriends parts
        # The largest substring is the first part when we split the word into numFriends parts
        # and take the first part as large as possible
        # So, the largest possible first part is the first (len(word) - (numFriends - 1)) characters
        # Because the remaining parts need to have at least one character each
        max_length = len(word) - (numFriends - 1)
        return word[:max_length]