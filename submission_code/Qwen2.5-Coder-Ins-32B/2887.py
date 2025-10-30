class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = "aeiouAEIOU"
        s_list = list(s)
        vowel_positions = [i for i, char in enumerate(s) if char in vowels]
        vowel_chars = sorted([s[i] for i in vowel_positions])
        
        for pos, char in zip(vowel_positions, vowel_chars):
            s_list[pos] = char
        
        return ''.join(s_list)