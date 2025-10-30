class Solution:
    def sortVowels(self, s: str) -> str:
        vowels_set = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
        vowels = [c for c in s if c in vowels_set]
        vowels.sort()
        result = list(s)
        ptr = 0
        for i in range(len(result)):
            if result[i] in vowels_set:
                result[i] = vowels[ptr]
                ptr += 1
        return ''.join(result)