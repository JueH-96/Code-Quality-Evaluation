class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        result = 0
        
        def has_all_vowels(s):
            return vowels.issubset(set(s))
        
        for i in range(n):
            consonant_count = 0
            for j in range(i, n):
                if word[j] not in vowels:
                    consonant_count += 1
                if consonant_count > k:
                    break
                if has_all_vowels(word[i:j+1]) and consonant_count == k:
                    result += 1
        
        return result