class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        count = 0
        
        for i in range(n):
            vowel_count = 0
            consonant_count = 0
            current_vowels = set()
            for j in range(i, n):
                if word[j] in vowels:
                    current_vowels.add(word[j])
                    vowel_count += 1
                else:
                    consonant_count += 1
                
                if consonant_count > k:
                    break
                
                if len(current_vowels) == 5 and consonant_count == k:
                    count += 1
        
        return count