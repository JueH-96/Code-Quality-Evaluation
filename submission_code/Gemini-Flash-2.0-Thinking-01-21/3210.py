class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set(['a', 'e', 'i', 'o', 'u'])
        count = 0
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                vowel_count = 0
                consonant_count = 0
                for char in substring:
                    if char in vowels:
                        vowel_count += 1
                    else:
                        consonant_count += 1
                if vowel_count == consonant_count:
                    if (vowel_count * consonant_count) % k == 0:
                        count += 1
        return count