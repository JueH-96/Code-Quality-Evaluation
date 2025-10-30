class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        vowel_to_bit = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
        count = 0
        n = len(word)
        
        for i in range(n):
            mask = 0
            consonant_count = 0
            for j in range(i, n):
                if word[j] in vowels:
                    mask |= (1 << vowel_to_bit[word[j]])
                else:
                    consonant_count += 1
                if mask == 31 and consonant_count == k:
                    count += 1
        return count