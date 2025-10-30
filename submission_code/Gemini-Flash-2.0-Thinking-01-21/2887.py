class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = "aeiouAEIOU"
        vowel_chars = []
        vowel_indices = []
        for index, char in enumerate(s):
            if char in vowels:
                vowel_chars.append(char)
                vowel_indices.append(index)
        vowel_chars.sort()
        s_list = list(s)
        for i in range(len(vowel_indices)):
            s_list[vowel_indices[i]] = vowel_chars[i]
        return "".join(s_list)