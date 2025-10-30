class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        count = 0
        vowels = "aeiou"

        for i in range(n):
            for j in range(i, n):
                substring = word[i:j+1]
                vowel_count = 0
                consonant_count = 0
                vowel_present = {'a': False, 'e': False, 'i': False, 'o': False, 'u': False}

                for char in substring:
                    if char in vowels:
                        vowel_present[char] = True
                        vowel_count += 1
                    else:
                        consonant_count += 1

                all_vowels_present = all(vowel_present.values())

                if all_vowels_present and consonant_count == k:
                    count += 1

        return count