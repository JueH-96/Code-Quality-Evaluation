class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        def is_vowel(c):
            return c in 'aeiou'
        
        n = len(s)
        count = 0
        
        for i in range(n):
            vowels = 0
            consonants = 0
            for j in range(i, n):
                if is_vowel(s[j]):
                    vowels += 1
                else:
                    consonants += 1
                
                if vowels == consonants and (vowels * consonants) % k == 0:
                    count += 1
        
        return count