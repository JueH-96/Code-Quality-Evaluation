class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False

        has_vowel = False
        has_consonant = False
        only_letters_digits = True

        vowels = "aeiouAEIOU"

        for char in word:
            if 'a' <= char <= 'z' or 'A' <= char <= 'Z' or '0' <= char <= '9':
                if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                    if char in vowels:
                        has_vowel = True
                    else:
                        has_consonant = True
            else:
                only_letters_digits = False
                break

        if not only_letters_digits:
            return False

        return has_vowel and has_consonant