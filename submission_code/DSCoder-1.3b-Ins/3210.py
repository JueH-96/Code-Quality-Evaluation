class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        count = 0
        for i in range(len(s)):
            vowels = consonants = 0
            for j in range(i, len(s)):
                if s[j] in 'aeiou':
                    vowels += 1
                else:
                    consonants += 1
                if vowels == consonants and (vowels * consonants) % k == 0:
                    count += 1
        return count