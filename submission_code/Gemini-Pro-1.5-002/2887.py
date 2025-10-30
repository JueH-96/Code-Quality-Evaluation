class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = "aeiouAEIOU"
        vowel_list = []
        result = ""
        for char in s:
            if char in vowels:
                vowel_list.append(char)
        vowel_list.sort()
        vowel_index = 0
        for char in s:
            if char in vowels:
                result += vowel_list[vowel_index]
                vowel_index += 1
            else:
                result += char
        return result