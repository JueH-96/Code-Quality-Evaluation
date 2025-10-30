class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels_set = {'a', 'e', 'i', 'o', 'u'}
        n = len(s)
        beautiful_count = 0
        
        # Iterate over all possible starting points of substrings
        for start in range(n):
            vowels = 0
            consonants = 0
            
            # Iterate over all possible ending points of substrings starting from 'start'
            for end in range(start, n):
                if s[end] in vowels_set:
                    vowels += 1
                else:
                    consonants += 1
                
                # Check if the current substring is beautiful
                if vowels == consonants and (vowels * consonants) % k == 0:
                    beautiful_count += 1
        
        return beautiful_count