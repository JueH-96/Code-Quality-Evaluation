class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        def has_all_vowels(counts):
            return all(counts[v] > 0 for v in 'aeiou')

        vowels = set('aeiou')
        n = len(word)
        result = 0
        
        for i in range(n):
            vowel_counts = {v: 0 for v in 'aeiou'}
            consonant_count = 0
            
            for j in range(i, n):
                if word[j] in vowels:
                    vowel_counts[word[j]] += 1
                else:
                    consonant_count += 1
                
                if consonant_count > k:
                    break
                
                if consonant_count == k and has_all_vowels(vowel_counts):
                    result += 1
        
        return result