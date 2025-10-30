class Solution:
    def sortVowels(self, s: str) -> str:
        vowel_set = set("aeiouAEIOU")
        vowels_list = [char for char in s if char in vowel_set]
        sorted_vowels = sorted(vowels_list)
        result = []
        idx = 0
        for char in s:
            if char in vowel_set:
                result.append(sorted_vowels[idx])
                idx += 1
            else:
                result.append(char)
        return ''.join(result)