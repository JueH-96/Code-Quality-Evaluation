class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        count = 0
        n = len(s)
        
        # Check all possible substrings
        for i in range(n):
            vowel_count = 0
            consonant_count = 0
            
            for j in range(i, n):
                # Update counts for current character
                if s[j] in vowels:
                    vowel_count += 1
                else:
                    consonant_count += 1
                
                # Check if current substring is beautiful
                if vowel_count == consonant_count:
                    # Since vowel_count == consonant_count, 
                    # the product is vowel_count * vowel_count
                    if (vowel_count * vowel_count) % k == 0:
                        count += 1
        
        return count