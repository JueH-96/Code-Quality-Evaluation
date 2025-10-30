class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels_set = {'a', 'e', 'i', 'o', 'u'}
        n = len(s)
        count = 0
        
        for start in range(n):
            vowels = 0
            consonants = 0
            
            for end in range(start, n):
                if s[end] in vowels_set:
                    vowels += 1
                else:
                    consonants += 1
                
                if vowels == consonants and (vowels * consonants) % k == 0:
                    count += 1
        
        return count