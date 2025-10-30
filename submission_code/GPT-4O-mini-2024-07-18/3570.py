class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        count = 0
        
        for start in range(n):
            consonant_count = 0
            vowel_set = set()
            
            for end in range(start, n):
                if word[end] in vowels:
                    vowel_set.add(word[end])
                else:
                    consonant_count += 1
                
                if len(vowel_set) == 5 and consonant_count == k:
                    count += 1
        
        return count