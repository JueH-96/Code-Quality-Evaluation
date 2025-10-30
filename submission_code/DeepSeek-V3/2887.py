class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set(['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'])
        s_list = list(s)
        vowel_indices = [i for i, char in enumerate(s_list) if char in vowels]
        vowel_chars = [s_list[i] for i in vowel_indices]
        vowel_chars_sorted = sorted(vowel_chars, key=lambda x: ord(x))
        for idx, char in zip(vowel_indices, vowel_chars_sorted):
            s_list[idx] = char
        return ''.join(s_list)