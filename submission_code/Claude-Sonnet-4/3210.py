class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        count = 0
        n = len(s)
        
        # Try all possible starting positions
        for i in range(n):
            vowel_count = 0
            consonant_count = 0
            
            # Extend substring from position i
            for j in range(i, n):
                # Add current character to counts
                if s[j] in vowels:
                    vowel_count += 1
                else:
                    consonant_count += 1
                
                # Check if current substring is beautiful
                if vowel_count == consonant_count:
                    if (vowel_count * consonant_count) % k == 0:
                        count += 1
        
        return count