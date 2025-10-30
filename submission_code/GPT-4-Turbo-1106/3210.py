class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(s)
        beautiful_count = 0
        
        for i in range(n):
            vowel_count = consonant_count = 0
            for j in range(i, n):
                if s[j] in vowels:
                    vowel_count += 1
                else:
                    consonant_count += 1
                
                if vowel_count == consonant_count and (vowel_count * consonant_count) % k == 0:
                    beautiful_count += 1
        
        return beautiful_count