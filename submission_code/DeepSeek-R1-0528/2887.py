class Solution:
    def sortVowels(self, s: str) -> str:
        vowels_set = set("AEIOUaeiou")
        s_list = list(s)
        vowels_list = [char for char in s_list if char in vowels_set]
        vowels_list.sort()
        vowel_ptr = 0
        for i in range(len(s_list)):
            if s_list[i] in vowels_set:
                s_list[i] = vowels_list[vowel_ptr]
                vowel_ptr += 1
        return "".join(s_list)