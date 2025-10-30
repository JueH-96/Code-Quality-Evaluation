class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = []
        consonants = []
        vowel_indices = []

        for i, char in enumerate(s):
            if char in 'aeiouAEIOU':
                vowels.append(char)
                vowel_indices.append(i)
            else:
                consonants.append(char)

        vowels.sort()

        result = list(s)
        for i, vowel in enumerate(vowels):
            result[vowel_indices[i]] = vowel

        return "".join(result)