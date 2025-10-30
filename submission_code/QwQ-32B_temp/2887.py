class Solution:
    def sortVowels(self, s: str) -> str:
        vowels_set = set("aeiouAEIOU")
        vowels_list = [c for c in s if c in vowels_set]
        sorted_vowels = sorted(vowels_list, key=lambda x: ord(x))
        res = []
        ptr = 0
        for char in s:
            if char in vowels_set:
                res.append(sorted_vowels[ptr])
                ptr += 1
            else:
                res.append(char)
        return ''.join(res)