class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        # Gather all lowercase letters in the word
        lower_set = {c for c in word if c.islower()}
        # Gather all uppercase letters, converted to lowercase
        upper_set = {c.lower() for c in word if c.isupper()}
        # A letter is special if it appears in both sets
        return len(lower_set & upper_set)