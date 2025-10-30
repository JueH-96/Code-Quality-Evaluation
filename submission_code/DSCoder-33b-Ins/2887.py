class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = [c for c in s if c in 'aeiouAEIOU']
        vowels.sort(key=str.lower)
        s_list = list(s)
        vowel_index = 0
        for i, c in enumerate(s_list):
            if c in 'aeiouAEIOU':
                s_list[i] = vowels[vowel_index]
                vowel_index += 1
        return ''.join(s_list)