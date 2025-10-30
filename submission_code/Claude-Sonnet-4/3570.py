class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        count = 0
        
        for i in range(n):
            vowel_set = set()
            consonant_count = 0
            
            for j in range(i, n):
                char = word[j]
                
                if char in vowels:
                    vowel_set.add(char)
                else:
                    consonant_count += 1
                
                # If we have too many consonants, break early
                if consonant_count > k:
                    break
                
                # Check if we have all 5 vowels and exactly k consonants
                if len(vowel_set) == 5 and consonant_count == k:
                    count += 1
        
        return count