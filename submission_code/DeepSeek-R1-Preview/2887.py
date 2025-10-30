class Solution:
    def sortVowels(self, s: str) -> str:
        s_list = list(s)
        vowels = []
        positions = []
        vowels_set = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
        for idx, c in enumerate(s_list):
            if c in vowels_set:
                vowels.append(c)
                positions.append(idx)
        vowels_sorted = sorted(vowels, key=lambda x: ord(x))
        for i, pos in enumerate(positions):
            s_list[pos] = vowels_sorted[i]
        return ''.join(s_list)