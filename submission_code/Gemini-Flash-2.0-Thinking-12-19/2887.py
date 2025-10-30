class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = "aeiouAEIOU"
        vowel_list = []
        s_list = list(s)
        vowel_indices = []

        for i in range(len(s_list)):
            if s_list[i] in vowels:
                vowel_list.append(s_list[i])
                vowel_indices.append(i)

        vowel_list.sort(key=ord)

        vowel_index_counter = 0
        for index in vowel_indices:
            s_list[index] = vowel_list[vowel_index_counter]
            vowel_index_counter += 1

        return "".join(s_list)