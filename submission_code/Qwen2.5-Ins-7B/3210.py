class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        count = 0
        for i in range(len(s)):
            vowels_count, consonants_count = 0, 0
            for j in range(i, len(s)):
                if s[j] in vowels:
                    vowels_count += 1
                else:
                    consonants_count += 1
                if vowels_count == consonants_count and (vowels_count * consonants_count) % k == 0:
                    count += 1
        return count