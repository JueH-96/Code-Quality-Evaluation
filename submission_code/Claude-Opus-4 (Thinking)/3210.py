class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        count = 0
        n = len(s)
        
        # Check all possible substrings
        for i in range(n):
            vowel_count = 0
            consonant_count = 0
            
            # Expand substring starting at position i
            for j in range(i, n):
                # Update counts based on current character
                if s[j] in vowels:
                    vowel_count += 1
                else:
                    consonant_count += 1
                
                # Check if current substring s[i:j+1] is beautiful
                if vowel_count == consonant_count and (vowel_count * consonant_count) % k == 0:
                    count += 1
        
        return count