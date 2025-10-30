class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        # Collect the lowercase letters present in `word`
        lowercase_letters = {ch for ch in word if ch.islower()}
        # Collect the lowercase form of the uppercase letters present in `word`
        uppercase_letters = {ch.lower() for ch in word if ch.isupper()}
        # The intersection of these two sets gives the letters that appear in both cases
        return len(lowercase_letters & uppercase_letters)