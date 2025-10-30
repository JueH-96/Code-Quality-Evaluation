class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        count = 0
        vowels = {'a', 'e', 'i', 'o', 'u'}
        
        for i in range(len(word)):
            vowel_present = {v: False for v in vowels}
            consonant_count = 0
            
            for j in range(i, len(word)):
                if word[j] in vowels:
                    vowel_present[word[j]] = True
                else:
                    consonant_count += 1
                
                if consonant_count > k:
                    break
                
                if consonant_count == k and all(vowel_present.values()):
                    count += 1
        
        return count