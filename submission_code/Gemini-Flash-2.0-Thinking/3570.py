class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        count = 0
        vowels = {'a', 'e', 'i', 'o', 'u'}

        for i in range(n):
            found_vowels = set()
            consonant_count = 0
            for j in range(i, n):
                char = word[j]
                if char in vowels:
                    found_vowels.add(char)
                elif 'a' <= char <= 'z':
                    consonant_count += 1

                if len(found_vowels) == 5 and consonant_count == k:
                    count += 1

        return count