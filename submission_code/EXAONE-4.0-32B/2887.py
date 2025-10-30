class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = "aeiouAEIOU"
        vowel_list = [char for char in s if char in vowels]
        vowel_list.sort()
        
        s_list = list(s)
        idx = 0
        for i in range(len(s)):
            if s[i] in vowels:
                s_list[i] = vowel_list[idx]
                idx += 1
                
        return ''.join(s_list)