class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        count = 0
        for i in range(len(word)):
            for j in range(i, len(word)):
                substring = word[i:j + 1]
                if len(substring) < 5:
                    continue
                vowel_count = 0
                consonant_count = 0
                has_all_vowels = True
                vowel_present = {'a': False, 'e': False, 'i': False, 'o': False, 'u': False}
                for char in substring:
                    if char in vowels:
                        vowel_count += 1
                        vowel_present[char] = True
                    else:
                        consonant_count += 1
                for vowel in vowels:
                    if not vowel_present[vowel]:
                        has_all_vowels = False
                        break
                if has_all_vowels and consonant_count == k:
                    count += 1
        return count