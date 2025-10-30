class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        count = 0
        n = len(word)
        
        for i in range(n):
            vowel_set = set()
            consonant_count = 0
            for j in range(i, n):
                if word[j] in vowels:
                    vowel_set.add(word[j])
                else:
                    consonant_count += 1
                if len(vowel_set) == 5 and consonant_count == k:
                    count += 1
        return count