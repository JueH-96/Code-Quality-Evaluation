class Solution:
    def sortVowels(self, s: str) -> str:
        vowels_set = {'a', 'e', 'i', 'o', 'u'}
        vowels = []
        # Collect all vowels from the string
        for c in s:
            if c.lower() in vowels_set:
                vowels.append(c)
        # Sort the vowels by their ASCII values
        sorted_vowels = sorted(vowels, key=lambda x: ord(x))
        # Build the result string
        res = []
        ptr = 0
        for c in s:
            if c.lower() in vowels_set:
                res.append(sorted_vowels[ptr])
                ptr += 1
            else:
                res.append(c)
        return ''.join(res)