class Solution:
    def sortVowels(self, s: str) -> str:
        vowels_set = {'a', 'e', 'i', 'o', 'u'}
        vowels_list = []
        # Collect all vowels from the string
        for c in s:
            if c.lower() in vowels_set:
                vowels_list.append(c)
        # Sort the collected vowels by their ASCII values
        sorted_vowels = sorted(vowels_list, key=lambda x: ord(x))
        # Build the result string
        res = []
        idx = 0
        for c in s:
            if c.lower() in vowels_set:
                res.append(sorted_vowels[idx])
                idx += 1
            else:
                res.append(c)
        return ''.join(res)