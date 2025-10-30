class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(s)
        count = 0
        
        for i in range(n):
            vowel_count = 0
            consonant_count = 0
            for j in range(i, n):
                if s[j] in vowels:
                    vowel_count += 1
                else:
                    consonant_count += 1
                
                # Check if this substring is beautiful
                if vowel_count == consonant_count and (vowel_count * consonant_count) % k == 0:
                    count += 1
        
        return count