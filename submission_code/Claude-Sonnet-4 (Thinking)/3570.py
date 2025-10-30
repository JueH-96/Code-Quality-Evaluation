class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        count = 0
        
        for i in range(n):
            vowel_set = set()
            consonant_count = 0
            
            for j in range(i, n):
                if word[j] in vowels:
                    vowel_set.add(word[j])
                else:
                    consonant_count += 1
                
                # Check if we have all vowels and exactly k consonants
                if len(vowel_set) == 5 and consonant_count == k:
                    count += 1
                
                # If we have more than k consonants, no point in extending further
                if consonant_count > k:
                    break
        
        return count