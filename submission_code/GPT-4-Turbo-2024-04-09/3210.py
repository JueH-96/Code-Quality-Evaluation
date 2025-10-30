class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(s)
        beautiful_count = 0
        
        # Check all possible substrings
        for start in range(n):
            vowel_count = 0
            consonant_count = 0
            for end in range(start, n):
                if s[end] in vowels:
                    vowel_count += 1
                else:
                    consonant_count += 1
                
                # Check if the current substring is beautiful
                if vowel_count == consonant_count:
                    if (vowel_count * consonant_count) % k == 0:
                        beautiful_count += 1
        
        return beautiful_count