class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set('aeiouAEIOU')
        # Collect and sort the vowels from the string
        vowel_list = sorted([c for c in s if c in vowels])
        vowel_idx = 0
        t = []
        for c in s:
            if c in vowels:
                t.append(vowel_list[vowel_idx])
                vowel_idx += 1
            else:
                t.append(c)
        return ''.join(t)