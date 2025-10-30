class Solution:
    def sortVowels(self, s: str) -> str:
        vowels_set = set('aeiouAEIOU')
        positions = []
        vowels = []
        for i, c in enumerate(s):
            if c in vowels_set:
                positions.append(i)
                vowels.append(c)
        vowels.sort()
        s_list = list(s)
        for pos, vowel in zip(positions, vowels):
            s_list[pos] = vowel
        return ''.join(s_list)